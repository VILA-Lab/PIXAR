"""
test_parallel.py — Multi-GPU parallel evaluation for PIXAR model.

Splits the test set into N equal chunks and evaluates each chunk on a
separate GPU in parallel. Raw intermediate counts from each worker are
merged and all final metrics are recomputed exactly — identical to running
test.py on the full set.

Usage:
    python test_parallel.py \\
      --version /path/to/model \\
      --dataset_dir /path/to/dataset \\
      --vision_pretrained /path/to/sam.pth \\
      --gpus 2,3,4,5 \\
      --output_dir ./evaluation/logs/my_eval_parallel \\
      [--seg_prompt_mode fuse] [--precision bf16] [--save_generated_text]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PIXAR Parallel Evaluation (Multi-GPU)"
    )
    parser.add_argument("--version", required=True, type=str,
                        help="Path to merged model (base + finetune weights)")
    parser.add_argument("--precision", default="fp16", type=str,
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--split", default="validation", type=str)
    parser.add_argument("--output_dir", default="./test_output_parallel", type=str)
    parser.add_argument("--workers", default=4, type=int)

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"])

    parser.add_argument("--num_obj_classes", type=int, default=81)
    parser.add_argument("--obj_threshold", type=float, default=0.5)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_generated_text", action="store_true", default=False)
    parser.add_argument("--text_output_file", type=str, default="generated_texts.json")

    parser.add_argument("--seg_prompt_mode", type=str, default="fuse",
                        choices=["seg_only", "text_only", "fuse"])
    parser.add_argument("--generate_text_in_seg_only", action="store_true", default=False,
                        help="Generate text tokens even in seg_only mode (default: disabled)")

    # Parallel-specific
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs, e.g. '2,3,4,5'")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _arr(x):
    """Convert numpy array or uninitialized scalar to JSON-serializable list."""
    if hasattr(x, "tolist"):
        return x.tolist()
    # AverageMeter.sum starts as int 0 when no tampered samples seen
    return [0.0, 0.0]


# ---------------------------------------------------------------------------
# Worker: runs in a spawned subprocess, one per GPU
# ---------------------------------------------------------------------------

def evaluate_worker(gpu, chunk_id, num_chunks, args, output_dir):
    """
    Load the model on `gpu`, evaluate indices [start, end), and save
    raw intermediate counts to output_dir/raw_chunk_{chunk_id}.json.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Local imports here so each spawned process initialises CUDA cleanly
    import tqdm
    import transformers
    from model.PIXAR import PIXARForCausalLM
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from utils.PIXAR_Set import CustomDataset
    from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                             AverageMeter, Summary, intersectionAndUnionGPU)

    print(f"[Chunk {chunk_id}] GPU {gpu}: loading tokenizer...", flush=True)

    # ---- Tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # ---- Model ----
    torch_dtype = {"fp16": torch.half, "bf16": torch.bfloat16}.get(
        args.precision, torch.float32
    )
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "obj_token_idx": args.obj_token_idx,
        "num_obj_classes": args.num_obj_classes,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_prompt_mode": args.seg_prompt_mode,
    }
    model = PIXARForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch_dtype)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    print(f"[Chunk {chunk_id}] GPU {gpu}: model loaded.", flush=True)

    # ---- Dataset ----
    test_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split=args.split,
        precision=args.precision,
        image_size=args.image_size,
    )

    # ---- Chunk index range ----
    import random
    all_indices = list(range(len(test_dataset)))
    random.seed(42)          # fixed seed → every worker gets the same shuffle
    random.shuffle(all_indices)
    chunk_size = (len(all_indices) + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(all_indices))
    indices = all_indices[start:end]
    print(
        f"[Chunk {chunk_id}] GPU {gpu}: indices {start}~{end-1} "
        f"({len(indices)}/{len(all_indices)} samples)",
        flush=True,
    )

    # ---- Default prompt ----
    default_prompt = (
        "Can you identify whether this image is real, fully synthetic, or tampered? "
        "If it is tampered, please (1) classify which object was modified and "
        "(2) output a mask for the modified regions."
    )

    # ---- Metric accumulators ----
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device="cpu")
    correct = 0
    total = 0

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    obj_tp_total = 0.0; obj_fp_total = 0.0; obj_fn_total = 0.0
    obj_exact_match_total = 0; obj_rows_total = 0
    obj_tp_per_class = None; obj_fp_per_class = None; obj_fn_per_class = None
    obj_hit1_total = 0; obj_hit5_total = 0; obj_hit_den_total = 0

    pix_TP = 0; pix_FP = 0; pix_FN = 0

    BINS = 512
    pos_hist = torch.zeros(BINS, device="cuda", dtype=torch.float64)
    neg_hist = torch.zeros(BINS, device="cuda", dtype=torch.float64)

    # ---- Real-time text output file ----
    gt_path = os.path.join(output_dir, f"generated_texts_chunk_{chunk_id}.jsonl")
    gt_file = open(gt_path, "w", encoding="utf-8") if args.save_generated_text else None

    # ---- Evaluation loop ----
    for sample_idx in tqdm.tqdm(indices, desc=f"GPU{gpu} chunk{chunk_id}"):
        item = test_dataset[sample_idx]
        (image_path, image, image_clip, conversations, mask, soft_mask,
         labels, cls_labels, resize, _, _, _, has_text, obj_label_vec) = item

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + default_prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "[CLS] [OBJ] [SEG] ")
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        image_clip = image_clip.unsqueeze(0).cuda()
        image = image.unsqueeze(0).cuda()
        if args.precision == "fp16":
            image_clip = image_clip.half(); image = image.half()
        elif args.precision == "bf16":
            image_clip = image_clip.bfloat16(); image = image.bfloat16()

        resize_list = [resize]
        original_size_list = [labels.shape[-2:]]

        generate_text = (
            args.seg_prompt_mode != "seg_only"
            or args.generate_text_in_seg_only
        )
        with torch.no_grad():
            output_ids, pred_masks, obj_preds, cls_info = model.evaluate(
                image_clip, image, input_ids, resize_list, original_size_list,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                cls_label=cls_labels,
                generate_text=generate_text,
            )

        # Decode text
        input_token_len = input_ids.shape[1]
        new_tokens = output_ids[0][input_token_len:]
        new_tokens = new_tokens[new_tokens != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
        text_output = text_output.replace("\n", " ").replace("  ", " ").strip()

        if cls_labels == 0:
            gt_text_description = ""
        elif cls_labels == 1:
            gt_text_description = ""
        else:
            conv_str = conversations[0]
            seg_marker = "[SEG] "
            seg_pos = conv_str.find(seg_marker)
            if seg_pos >= 0:
                gt_text_description = conv_str[seg_pos + len(seg_marker):].split("</s>")[0].strip()
                hardcoded_prefix = "The image is tampered."
                if gt_text_description.startswith(hardcoded_prefix):
                    remaining = gt_text_description[len(hardcoded_prefix):].strip()
                    gt_text_description = (
                        f"This image is tampered. {remaining}" if remaining else ""
                    )
            else:
                gt_text_description = ""

        if gt_file is not None:
            gt_file.write(json.dumps({
                "image_path": image_path,
                "generated_text": text_output,
                "gt_text_description": gt_text_description,
                "ground_truth_label": int(cls_labels),
                "predicted_class": cls_info["predicted_class"],
                "predicted_label": cls_info["label"],
            }, ensure_ascii=False) + "\n")
            gt_file.flush()

        # ------ Classification ------
        predicted_class = cls_info["predicted_class"]
        preds  = torch.tensor([predicted_class], device="cuda")
        gt_cls = torch.tensor([cls_labels],      device="cuda")
        correct += (preds == gt_cls).sum().item()
        total += 1
        confusion_matrix[int(cls_labels), predicted_class] += 1

        # ------ Segmentation (tampered only) ------
        if cls_labels == 2:
            gt_mask      = soft_mask.int().cuda()
            pred_mask_bin = (pred_masks[0] > 0).int().cuda()

            intersection = union = acc_iou = 0.0
            for mask_i, output_i in zip(gt_mask, pred_mask_bin):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0
            intersection = intersection.cpu().numpy()
            union        = union.cpu().numpy()
            acc_iou      = acc_iou.cpu().numpy() / gt_mask.shape[0]
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=gt_mask.shape[0])

            with torch.no_grad():
                pm = pred_masks[0].float().cuda()
                pred_scores = (
                    torch.sigmoid(pm) if (pm.min() < 0 or pm.max() > 1.0)
                    else pm.clamp(0, 1)
                )
            pred_bin = (pred_scores >= 0.5).to(torch.int32)

            for mask_i, score_i, bin_i in zip(gt_mask, pred_scores, pred_bin):
                m_flat = mask_i.flatten().to(torch.uint8)
                p_flat = bin_i.flatten().to(torch.uint8)
                s_flat = score_i.flatten().to(torch.float32)

                pix_TP += (p_flat.eq(1) & m_flat.eq(1)).sum().item()
                pix_FP += (p_flat.eq(1) & m_flat.eq(0)).sum().item()
                pix_FN += (p_flat.eq(0) & m_flat.eq(1)).sum().item()

                s_clamped = s_flat.clamp_(0, 1)
                bins_idx  = torch.clamp((s_clamped * (BINS - 1)).long(), 0, BINS - 1)
                m_bool    = (m_flat > 0)
                if m_bool.any():
                    pos_hist.index_add_(
                        0, bins_idx[m_bool],
                        torch.ones_like(bins_idx[m_bool], dtype=torch.float64)
                    )
                if (~m_bool).any():
                    neg_hist.index_add_(
                        0, bins_idx[~m_bool],
                        torch.ones_like(bins_idx[~m_bool], dtype=torch.float64)
                    )

        # ------ OBJ (tampered only) ------
        if cls_labels == 2:
            gt       = obj_label_vec.unsqueeze(0).cuda()
            probs_obj = obj_preds.unsqueeze(0) if obj_preds.dim() == 1 else obj_preds
            pred     = (probs_obj >= args.obj_threshold).to(gt.dtype)

            gt_bool   = (gt > 0).to(torch.bool)
            valid_rows = gt_bool.any(dim=1)
            n_valid   = int(valid_rows.sum().item())
            if n_valid > 0:
                K = gt.shape[1]; k5 = min(5, K)
                topk_idx = probs_obj.topk(k5, dim=1).indices
                top1_idx = topk_idx[:, :1]
                hit1 = gt_bool.gather(1, top1_idx).any(dim=1)
                topk_mask = torch.zeros_like(gt_bool)
                topk_mask.scatter_(1, topk_idx, True)
                hit5 = (topk_mask & gt_bool).any(dim=1)
                obj_hit1_total += int(hit1[valid_rows].sum().item())
                obj_hit5_total += int(hit5[valid_rows].sum().item())
                obj_hit_den_total += n_valid

            if obj_tp_per_class is None:
                K = gt.shape[1]
                obj_tp_per_class = torch.zeros(K, device="cuda", dtype=torch.float64)
                obj_fp_per_class = torch.zeros(K, device="cuda", dtype=torch.float64)
                obj_fn_per_class = torch.zeros(K, device="cuda", dtype=torch.float64)

            tp = (pred * gt).sum().double()
            fp = (pred * (1 - gt)).sum().double()
            fn = ((1 - pred) * gt).sum().double()
            obj_tp_total += tp.item(); obj_fp_total += fp.item(); obj_fn_total += fn.item()
            obj_exact_match_total += (pred == gt).all(dim=1).sum().item()
            obj_rows_total        += gt.shape[0]
            obj_tp_per_class += (pred * gt).sum(dim=0).double()
            obj_fp_per_class += (pred * (1 - gt)).sum(dim=0).double()
            obj_fn_per_class += ((1 - pred) * gt).sum(dim=0).double()

    # ---- Save raw counts ----
    raw = {
        "confusion_matrix":       confusion_matrix.tolist(),
        "correct":                correct,
        "total":                  total,
        "intersection_sum":       _arr(intersection_meter.sum),
        "union_sum":              _arr(union_meter.sum),
        "acc_iou_sum":            _arr(acc_iou_meter.sum),
        "acc_iou_count":          int(acc_iou_meter.count),
        "pix_TP":                 int(pix_TP),
        "pix_FP":                 int(pix_FP),
        "pix_FN":                 int(pix_FN),
        "pos_hist":               pos_hist.cpu().tolist(),
        "neg_hist":               neg_hist.cpu().tolist(),
        "obj_tp_total":           obj_tp_total,
        "obj_fp_total":           obj_fp_total,
        "obj_fn_total":           obj_fn_total,
        "obj_exact_match_total":  int(obj_exact_match_total),
        "obj_rows_total":         int(obj_rows_total),
        "obj_hit1_total":         int(obj_hit1_total),
        "obj_hit5_total":         int(obj_hit5_total),
        "obj_hit_den_total":      int(obj_hit_den_total),
        "obj_tp_per_class":       obj_tp_per_class.cpu().tolist() if obj_tp_per_class is not None else None,
        "obj_fp_per_class":       obj_fp_per_class.cpu().tolist() if obj_fp_per_class is not None else None,
        "obj_fn_per_class":       obj_fn_per_class.cpu().tolist() if obj_fn_per_class is not None else None,
    }
    raw_path = os.path.join(output_dir, f"raw_chunk_{chunk_id}.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f)
    print(f"[Chunk {chunk_id}] Raw counts saved → {raw_path}", flush=True)

    if gt_file is not None:
        gt_file.close()
        print(f"[Chunk {chunk_id}] Generated texts saved → {gt_path}", flush=True)


# ---------------------------------------------------------------------------
# Merge: sum raw counts from all chunks
# ---------------------------------------------------------------------------

def merge_raw(raws):
    m = {}

    # Confusion matrix
    cm = np.array(raws[0]["confusion_matrix"], dtype=np.float64)
    for r in raws[1:]:
        cm += np.array(r["confusion_matrix"], dtype=np.float64)
    m["confusion_matrix"] = cm.tolist()
    m["correct"] = sum(r["correct"] for r in raws)
    m["total"]   = sum(r["total"]   for r in raws)

    # Segmentation meters
    inter = np.array(raws[0]["intersection_sum"], dtype=np.float64)
    union = np.array(raws[0]["union_sum"],         dtype=np.float64)
    acc_s = np.array(raws[0]["acc_iou_sum"],       dtype=np.float64)
    acc_c = raws[0]["acc_iou_count"]
    for r in raws[1:]:
        inter += np.array(r["intersection_sum"], dtype=np.float64)
        union += np.array(r["union_sum"],         dtype=np.float64)
        acc_s += np.array(r["acc_iou_sum"],       dtype=np.float64)
        acc_c += r["acc_iou_count"]
    m["intersection_sum"] = inter.tolist()
    m["union_sum"]        = union.tolist()
    m["acc_iou_sum"]      = acc_s.tolist()
    m["acc_iou_count"]    = acc_c

    # Pixel counts
    m["pix_TP"] = sum(r["pix_TP"] for r in raws)
    m["pix_FP"] = sum(r["pix_FP"] for r in raws)
    m["pix_FN"] = sum(r["pix_FN"] for r in raws)

    # AUC histograms
    pos_h = np.array(raws[0]["pos_hist"], dtype=np.float64)
    neg_h = np.array(raws[0]["neg_hist"], dtype=np.float64)
    for r in raws[1:]:
        pos_h += np.array(r["pos_hist"], dtype=np.float64)
        neg_h += np.array(r["neg_hist"], dtype=np.float64)
    m["pos_hist"] = pos_h.tolist()
    m["neg_hist"] = neg_h.tolist()

    # OBJ scalars
    for key in ("obj_tp_total", "obj_fp_total", "obj_fn_total",
                "obj_exact_match_total", "obj_rows_total",
                "obj_hit1_total", "obj_hit5_total", "obj_hit_den_total"):
        m[key] = sum(r[key] for r in raws)

    # OBJ per-class vectors
    tp_c = fp_c = fn_c = None
    for r in raws:
        if r["obj_tp_per_class"] is not None:
            t  = np.array(r["obj_tp_per_class"], dtype=np.float64)
            f_ = np.array(r["obj_fp_per_class"], dtype=np.float64)
            fn = np.array(r["obj_fn_per_class"], dtype=np.float64)
            if tp_c is None:
                tp_c, fp_c, fn_c = t, f_, fn
            else:
                tp_c += t; fp_c += f_; fn_c += fn
    m["obj_tp_per_class"] = tp_c.tolist() if tp_c is not None else None
    m["obj_fp_per_class"] = fp_c.tolist() if fp_c is not None else None
    m["obj_fn_per_class"] = fn_c.tolist() if fn_c is not None else None

    return m


# ---------------------------------------------------------------------------
# Compute and print final metrics from merged raw counts
# ---------------------------------------------------------------------------

def compute_and_print(m, num_chunks):
    num_classes = 3

    # Pixel P/R/F1
    pix_TP, pix_FP, pix_FN = m["pix_TP"], m["pix_FP"], m["pix_FN"]
    pixel_precision = pix_TP / (pix_TP + pix_FP + 1e-12) if (pix_TP + pix_FP) > 0 else 0.0
    pixel_recall    = pix_TP / (pix_TP + pix_FN + 1e-12) if (pix_TP + pix_FN) > 0 else 0.0
    pixel_f1        = (2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-12)
                       if (pixel_precision + pixel_recall) > 0 else 0.0)

    # ROC-AUC from histograms
    pos_hist = torch.tensor(m["pos_hist"], dtype=torch.float64)
    neg_hist = torch.tensor(m["neg_hist"], dtype=torch.float64)
    if (pos_hist.sum() + neg_hist.sum()) > 0:
        pos_cum = torch.cumsum(pos_hist.flip(0), dim=0)
        neg_cum = torch.cumsum(neg_hist.flip(0), dim=0)
        tp_h = pos_cum; fp_h = neg_cum
        P = pos_cum[-1]; N = neg_cum[-1]
        fn_h = P - tp_h; tn_h = N - fp_h
        precision_h = tp_h / (tp_h + fp_h + 1e-12)
        recall_h    = tp_h / (tp_h + fn_h + 1e-12)
        dr = recall_h[:-1] - recall_h[1:]
        pixel_pr_auc  = float(torch.sum(precision_h[1:] * dr).item())
        fpr = fp_h / (fp_h + tn_h + 1e-12)
        tpr = recall_h
        df  = fpr[1:] - fpr[:-1]
        pixel_roc_auc = float(torch.sum((tpr[1:] + tpr[:-1]) * 0.5 * df).item())
    else:
        pixel_pr_auc = pixel_roc_auc = 0.0

    # OBJ metrics
    obj_tp, obj_fp, obj_fn = m["obj_tp_total"], m["obj_fp_total"], m["obj_fn_total"]
    obj_micro_prec = obj_tp / (obj_tp + obj_fp + 1e-12) if (obj_tp + obj_fp) > 0 else 0.0
    obj_micro_rec  = obj_tp / (obj_tp + obj_fn + 1e-12) if (obj_tp + obj_fn) > 0 else 0.0
    obj_micro_f1   = (2 * obj_micro_prec * obj_micro_rec / (obj_micro_prec + obj_micro_rec + 1e-12)
                      if (obj_micro_prec + obj_micro_rec) > 0 else 0.0)
    obj_subset_acc = (m["obj_exact_match_total"] / m["obj_rows_total"]
                      if m["obj_rows_total"] > 0 else 0.0)
    obj_top1 = (m["obj_hit1_total"] / m["obj_hit_den_total"] * 100.0
                if m["obj_hit_den_total"] > 0 else 0.0)
    obj_top5 = (m["obj_hit5_total"] / m["obj_hit_den_total"] * 100.0
                if m["obj_hit_den_total"] > 0 else 0.0)

    if m["obj_tp_per_class"] is not None:
        tp_c = np.array(m["obj_tp_per_class"])
        fp_c = np.array(m["obj_fp_per_class"])
        fn_c = np.array(m["obj_fn_per_class"])
        prec_c = tp_c / (tp_c + fp_c + 1e-12)
        rec_c  = tp_c / (tp_c + fn_c + 1e-12)
        f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
        obj_macro_prec = float(prec_c.mean())
        obj_macro_rec  = float(rec_c.mean())
        obj_macro_f1   = float(f1_c.mean())
    else:
        obj_macro_prec = obj_macro_rec = obj_macro_f1 = 0.0

    # IoU
    inter     = np.array(m["intersection_sum"])
    union_arr = np.array(m["union_sum"])
    iou_class = inter / (union_arr + 1e-10)
    ciou      = float(iou_class[1]) if len(iou_class) > 1 else 0.0
    acc_iou_sum   = np.array(m["acc_iou_sum"])
    acc_iou_count = m["acc_iou_count"]
    giou = (float(acc_iou_sum[1] / acc_iou_count)
            if (acc_iou_count > 0 and len(acc_iou_sum) > 1) else 0.0)

    # Classification
    correct, total = m["correct"], m["total"]
    accuracy = correct / total * 100.0 if total > 0 else 0.0

    class_names = ["Real", "Full Synthetic", "Tampered"]
    cm = np.array(m["confusion_matrix"])
    per_class_metrics = {}
    for i in range(num_classes):
        tp_i  = cm[i, i]
        fp_i  = cm[:, i].sum() - tp_i
        fn_i  = cm[i, :].sum() - tp_i
        tot_i = cm[i, :].sum()
        prec_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        rec_i  = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        f1_i   = float(2 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0
        per_class_metrics[class_names[i]] = {
            "accuracy":  float(tp_i / tot_i) if tot_i > 0 else 0.0,
            "precision": prec_i,
            "recall":    rec_i,
            "f1":        f1_i,
        }

    iou      = ciou
    f1_score = (2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10)
                if (iou + accuracy / 100) > 0 else 0.0)

    # ---- Print ----
    print(f"\n{'='*70}")
    print(f"Parallel Test Results ({total} samples, {num_chunks} chunks merged)")
    print(f"{'='*70}")

    print(f"\nClassification Accuracy: {accuracy:.4f}%")
    print("\nPer-Class Metrics:")
    for cn, met in per_class_metrics.items():
        print(f"  {cn}:")
        print(f"    Accuracy:  {met['accuracy']:.4f}")
        print(f"    Precision: {met['precision']:.4f}")
        print(f"    Recall:    {met['recall']:.4f}")
        print(f"    F1 Score:  {met['f1']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"{'':20}", end="")
    for name in class_names:
        print(f"{name:>15}", end="")
    print()
    for i, cn in enumerate(class_names):
        print(f"{cn:20}", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:15.0f}", end="")
        print()

    print(f"\nSegmentation Metrics (tampered only):")
    print(f"  gIoU: {giou:.4f}")
    print(f"  cIoU: {ciou:.4f}")
    print(f"  Pixel Precision: {pixel_precision:.4f}")
    print(f"  Pixel Recall:    {pixel_recall:.4f}")
    print(f"  Pixel F1:        {pixel_f1:.4f}")
    print(f"  Pixel ROC-AUC:   {pixel_roc_auc:.4f}")

    print(f"\n[OBJ] Multi-Label Metrics (tampered only):")
    print(f"  Micro  - P: {obj_micro_prec:.4f}, R: {obj_micro_rec:.4f}, F1: {obj_micro_f1:.4f}")
    print(f"  Macro  - P: {obj_macro_prec:.4f}, R: {obj_macro_rec:.4f}, F1: {obj_macro_f1:.4f}")
    print(f"  Subset Acc: {obj_subset_acc:.4f}")
    print(f"  Top-1 Acc:  {obj_top1:.4f}%")
    print(f"  Top-5 Acc:  {obj_top5:.4f}%")

    print(f"\nCombined F1: {f1_score:.4f}")

    return {
        "accuracy":          accuracy,
        "giou":              giou,
        "ciou":              ciou,
        "pixel_precision":   pixel_precision,
        "pixel_recall":      pixel_recall,
        "pixel_f1":          pixel_f1,
        "pixel_roc_auc":     pixel_roc_auc,
        "obj_micro_f1":      obj_micro_f1,
        "obj_macro_f1":      obj_macro_f1,
        "obj_top1":          obj_top1,
        "obj_top5":          obj_top5,
        "per_class_metrics": per_class_metrics,
        "total_samples":     total,
        "combined_f1":       f1_score,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",")]
    num_chunks = len(gpus)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parallel evaluation: {num_chunks} chunks on GPUs {gpus}")
    print(f"Model:      {args.version}")
    print(f"Dataset:    {args.dataset_dir}")
    print(f"Output dir: {args.output_dir}")

    ctx = mp.get_context("spawn")
    procs = []
    for chunk_id, gpu in enumerate(gpus):
        p = ctx.Process(
            target=evaluate_worker,
            args=(gpu, chunk_id, num_chunks, args, args.output_dir),
        )
        p.start()
        procs.append(p)
        print(f"  Launched chunk {chunk_id} on GPU {gpu} (PID={p.pid})")

    print("Waiting for all chunks to finish...")
    failed = []
    for chunk_id, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            failed.append(chunk_id)
            print(
                f"  [ERROR] Chunk {chunk_id} (GPU {gpus[chunk_id]}) "
                f"failed with exitcode {p.exitcode}"
            )

    if failed:
        raise RuntimeError(
            f"Chunks {failed} failed. "
            f"Check {args.output_dir}/raw_chunk_*.json for which chunks completed."
        )

    # ---- Merge ----
    print("\nAll chunks done. Merging results...")
    raws = []
    for i in range(num_chunks):
        raw_path = os.path.join(args.output_dir, f"raw_chunk_{i}.json")
        with open(raw_path) as f:
            raws.append(json.load(f))

    merged  = merge_raw(raws)
    metrics = compute_and_print(merged, num_chunks)

    # Optionally merge generated text files (JSONL per chunk → single JSON)
    if args.save_generated_text:
        all_texts = []
        for i in range(num_chunks):
            gt_path = os.path.join(args.output_dir, f"generated_texts_chunk_{i}.jsonl")
            if os.path.exists(gt_path):
                with open(gt_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_texts.append(json.loads(line))
        out_path = os.path.join(args.output_dir, args.text_output_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_texts, f, indent=2, ensure_ascii=False)
        print(f"Generated texts saved to: {out_path}")

    # Save final metrics.json
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

