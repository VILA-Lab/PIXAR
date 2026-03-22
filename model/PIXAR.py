from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel, AutoTokenizer

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)

from .segment_anything import build_sam_vit_h

from torchviz import make_dot
import itertools

import deepspeed

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, eps: float = 1e-6):
    """
    inputs: logits from SAM (B,1,H,W 或兼容形状)
    targets: 二值/软掩码，取值应在 [0,1]
    """
    # 强制在 FP32 里计算，更稳
    probs   = inputs.float().sigmoid()      # logits -> prob
    probs   = probs.flatten(1, -1)          # 标准展平
    targets = torch.clamp(targets.float(), 0.0, 1.0).flatten(1, -1)

    numerator   = 2.0 * (probs * targets).sum(-1)
    denominator = (probs + targets).sum(-1).clamp_min(1e-3)
    loss = 1.0 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    # 在 FP32 中算更稳
    with torch.cuda.amp.autocast(enabled=False):
        logits = inputs.float()
        gt     = targets.float().clamp_(0.0, 1.0)
        per_pix = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")  # [N,1,H,W] 或 [N,H,W]

        # 统一展平到每样本的像素维：对所有像素取均值，再对样本求和
        per_pix = per_pix.view(per_pix.size(0), -1).mean(1).sum()

    return per_pix / (num_masks + 1e-8)



class PixarMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PixarMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_pixar_modules(self.config)

    def initialize_pixar_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        in_dim = config.hidden_size
        out_dim = config.out_dim

        # Classification head (3-way: real / fully synthetic / tampered)
        self.cls_head = nn.ModuleList([nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(in_dim // 2, 3),
        )])

        # Object recognition head (multi-label)
        self.obj_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(in_dim // 2, config.num_obj_classes),
        )

        # Gated SEG + description fusion for segmentation
        self.seg_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
        self.gate_mlp = nn.Linear(2 * out_dim, 1)

        # Set all trainable
        for module in [self.cls_head, self.obj_head, self.seg_proj, self.text_proj, self.gate_mlp]:
            module.train()
            for p in module.parameters():
                p.requires_grad = True

class PixarModel(PixarMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(PixarModel, self).__init__(config, **kwargs)
        
        print("\nInitializing PixarModel:")
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.config.vision_hidden_size = 256
        self.config.fc_hidden_size = 1408
        self.config.llm_input_size = 1024

class PIXARForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower

        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 1.0)
        self.cls_loss_weight = kwargs.pop("cls_loss_weight", 1.0)
        self.mask_loss_weight =  kwargs.pop("mask_loss_weight", 1.0)
        self.obj_loss_weight = kwargs.pop("obj_loss_weight", 1.0)
        self.text_loss_weight = kwargs.pop("text_loss_weight", 1.0)  # NEW: text loss weight
        self.obj_token_idx   = kwargs.pop("obj_token_idx", None)
        config.num_obj_classes = kwargs.pop("num_obj_classes", 81)
        
        # >>> 新增：pos_weight 配置 <<<
        self.fixed_obj_pos_weight = kwargs.pop("obj_pos_weight", None)
        self.obj_pos_weight_max   = kwargs.pop("obj_pos_weight_max", 100.0)

        # 2. Initialize base model
        self.cls_token_idx = kwargs.pop("cls_token_idx")
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)
        self.model = PixarModel(config, **kwargs)
        self.model.initialize_pixar_modules(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.seg_prompt_mode = kwargs.pop("seg_prompt_mode", "fuse")
        self.mask_type = kwargs.pop("mask_type", "ours")

        self.post_init()
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings
    
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    
    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        cls_labels: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        soft_masks_list: List[torch.FloatTensor],
        obj_labels: torch.FloatTensor,
        cls_labels_list: List[torch.LongTensor] = None,
        label_list: List[torch.Tensor] = None,
        resize_list: List[tuple] = None,
        inference: bool = False,
        **kwargs,
    ):
        if images.size(0) != images_clip.size(0):
            raise ValueError(f"Batch size mismatch: images {images.size(0)} != images_clip {images_clip.size(0)}")

        image_embeddings = self.get_visual_embs(images)
        B = image_embeddings.shape[0]
        assert B == len(offset) - 1

        # ===== LLM Forward Pass =====
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states = [output_hidden_states_level]
            text_loss = torch.tensor(0.0, device=images.device)
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

            # LM text loss — for ALL samples (all have text after [SEG])
            if hasattr(output, 'loss') and output.loss is not None and not torch.isnan(output.loss):
                text_loss = output.loss
            else:
                text_loss = torch.tensor(0.0, device=images.device)

        # ===== Clean offset-based token position extraction =====
        hs = output_hidden_states[-1]  # [B, T_expanded, H_dim]
        T_input = input_ids.shape[1]
        T_hidden = hs.shape[1]
        image_offset = T_hidden - T_input  # num_image_tokens - 1
        H_dim = hs.shape[-1]

        # Find structure token positions in input_ids
        cls_pos = (input_ids == self.cls_token_idx)
        obj_pos = (input_ids == self.obj_token_idx)
        seg_pos = (input_ids == self.seg_token_idx)

        assert cls_pos.sum(1).eq(1).all(), f"Each sample must have exactly one [CLS], got {cls_pos.sum(1)}"
        assert obj_pos.sum(1).eq(1).all(), f"Each sample must have exactly one [OBJ], got {obj_pos.sum(1)}"
        assert seg_pos.sum(1).eq(1).all(), f"Each sample must have exactly one [SEG], got {seg_pos.sum(1)}"

        # Extract hidden vectors at token positions (shifted by image_offset)
        cls_vec = torch.zeros(B, H_dim, device=hs.device, dtype=hs.dtype)
        obj_vec = torch.zeros(B, H_dim, device=hs.device, dtype=hs.dtype)
        seg_vec = torch.zeros(B, H_dim, device=hs.device, dtype=hs.dtype)
        seg_h_indices = []

        for b in range(B):
            c = cls_pos[b].nonzero()[0].item()
            o = obj_pos[b].nonzero()[0].item()
            s = seg_pos[b].nonzero()[0].item()
            cls_vec[b] = hs[b, c + image_offset]
            obj_vec[b] = hs[b, o + image_offset]
            seg_vec[b] = hs[b, s + image_offset]
            seg_h_indices.append(s + image_offset)

        # ===== Classification Head =====
        cls_logits = self.model.cls_head[0](cls_vec)  # [B, 3]
        cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_labels)

        # ===== Object Head (loss only for tampered) =====
        obj_logits = self.model.obj_head(obj_vec)  # [B, K]
        # Keep obj_head in the computation graph even when no real obj_loss is computed,
        # to prevent DeepSpeed ZeRO all-reduce deadlock across ranks.
        obj_loss = (obj_logits * 0.0).sum()
        tampered_mask = (cls_labels == 2)
        if tampered_mask.any() and obj_labels is not None and obj_labels.numel() > 0:
            tampered_obj_logits = obj_logits[tampered_mask]
            tampered_obj_labels = obj_labels[tampered_mask]
            n = min(tampered_obj_logits.size(0), tampered_obj_labels.size(0))
            if n > 0:
                if self.fixed_obj_pos_weight is not None:
                    pos_w = torch.full(
                        (tampered_obj_logits.size(1),),
                        float(self.fixed_obj_pos_weight),
                        device=tampered_obj_logits.device,
                        dtype=tampered_obj_logits.dtype,
                    )
                else:
                    with torch.no_grad():
                        p = tampered_obj_labels[:n].float().mean(dim=0)
                        pos_w = ((1.0 - p) / (p.clamp_min(1e-6))).clamp(1.0, self.obj_pos_weight_max)
                obj_loss = obj_loss + F.binary_cross_entropy_with_logits(
                    tampered_obj_logits[:n], tampered_obj_labels[:n].float(),
                    reduction="mean", pos_weight=pos_w,
                )

        # ===== Segmentation with gated SEG + description fusion (tampered only) =====
        mask_bce_loss = torch.tensor(0.0, device=cls_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=cls_loss.device)
        mask_loss = torch.tensor(0.0, device=cls_loss.device)
        pred_masks = []
        num_masks = 0

        if tampered_mask.any():
            tampered_indices = tampered_mask.nonzero(as_tuple=True)[0]
            for b in tampered_indices:
                b = b.item()
                # SEG embedding
                seg_emb = self.model.seg_proj(seg_vec[b])  # [out_dim]
                
                # Description span: from seg_h_idx+1 to end of real tokens
                seg_h_idx = seg_h_indices[b]
                seq_len = int(attention_masks[b].sum().item())
                end_h_idx = seq_len + image_offset
                desc_h = hs[b, seg_h_idx + 1:end_h_idx, :]

                if desc_h.shape[0] > 0:
                    text_vec_b = desc_h.mean(dim=0)
                else:
                    text_vec_b = torch.zeros(H_dim, device=hs.device, dtype=hs.dtype)
                text_emb = self.model.text_proj(text_vec_b)  # [out_dim]

                # Gated fusion
                mode = getattr(self, "seg_prompt_mode", "fuse")

                # 只有需要 text 的模式才算 text（避免无谓计算）
                text_emb = None
                if mode in ["fuse", "text_only"]:
                    seg_h_idx = seg_h_indices[b]
                    seq_len = int(attention_masks[b].sum().item())
                    end_h_idx = seq_len + image_offset
                    desc_h = hs[b, seg_h_idx + 1:end_h_idx, :]

                    if desc_h.shape[0] > 0:
                        text_vec_b = desc_h.mean(dim=0)
                    else:
                        text_vec_b = torch.zeros(H_dim, device=hs.device, dtype=hs.dtype)

                    text_emb = self.model.text_proj(text_vec_b)  # [out_dim]

                # === Ablation switch ===
                if mode == "seg_only":
                    fused = seg_emb
                elif mode == "text_only":
                    fused = text_emb
                else:
                    # fuse
                    gate = torch.sigmoid(self.model.gate_mlp(torch.cat([seg_emb, text_emb], dim=-1)))
                    fused = gate * seg_emb + (1 - gate) * text_emb


                # Feed to SAM
                text_embeds = fused.unsqueeze(0).unsqueeze(1)  # [1, 1, out_dim]
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None, boxes=None, masks=None, text_embeds=text_embeds,
                )
                sparse_embeddings = sparse_embeddings.to(fused.dtype)
                low_res_masks, _ = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[b].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[b],
                    original_size=label_list[b].shape,
                )
                pred_masks.append(pred_mask[:, 0])

                # Compute mask loss
                gt_mask = masks_list[b]
                gt_soft_mask = soft_masks_list[b]
                target_mask = gt_soft_mask if self.mask_type == "ours" else gt_mask
                assert target_mask.shape[0] == pred_mask[:, 0].shape[0], \
                    f"target_mask.shape: {target_mask.shape}, pred_mask.shape: {pred_mask[:, 0].shape}"

                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask[:, 0], target_mask, num_masks=target_mask.shape[0])
                    * target_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask[:, 0], target_mask, num_masks=target_mask.shape[0])
                    * target_mask.shape[0]
                )
                num_masks += target_mask.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

        # Dummy loss to keep unused trainable params in the graph when no tampered samples.
        # Without this, DeepSpeed ZeRO all-reduce deadlocks because some ranks have
        # gradients for these params while others don't.
        # Note: obj_head is always in the graph via obj_loss = (obj_logits * 0.0).sum()
        if not inference and not tampered_mask.any():
            dummy = torch.zeros([], device=cls_loss.device)
            for p in itertools.chain(
                self.model.visual_model.mask_decoder.parameters(),
                self.model.seg_proj.parameters(),
                self.model.text_proj.parameters(),
                self.model.gate_mlp.parameters(),
            ):
                dummy = dummy + p.sum() * 0.0
            mask_loss = mask_loss + dummy

        # ===== Total Loss =====
        loss = (
            self.mask_loss_weight * mask_loss
            + self.cls_loss_weight * cls_loss
            + self.obj_loss_weight * obj_loss
            + self.text_loss_weight * text_loss
        )

        # ===== Return =====
        if inference:
            out = {
                "logits": cls_logits,
                "obj_logits": obj_logits,
            }
            if tampered_mask.any():
                out.update({
                    "pred_masks": pred_masks,
                    "gt_masks": masks_list,
                    "gt_soft_masks": soft_masks_list,
                })
            return out

        return {
            "loss": loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "cls_loss": cls_loss,
            "obj_loss": obj_loss,
            "text_loss": text_loss,
            "logits": cls_logits,
        }
        
    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=64,
        tokenizer=None,
        cls_label=None,
        generate_text=False,
    ):
        """
        Two-stage inference with early exit for non-tampered samples:
        Stage 1: Cheap forward pass on input_ids to classify (no generation).
        Stage 2 (tampered only): Generate text + segment.
        """
        with torch.no_grad():
            # ── Stage 1: Forward pass on input_ids (no generation) ──
            was_training = self.training
            self.train()
            fwd_output = super(PIXARForCausalLM, self).forward(
                images=images_clip,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            if not was_training:
                self.eval()

            hs = fwd_output.hidden_states[-1]  # [1, T_expanded, H]
            T_input = input_ids.shape[1]
            T_hidden = hs.shape[1]
            image_offset = T_hidden - T_input
            H_dim = hs.shape[-1]

            # Find structure token positions
            cls_idx = (input_ids[0] == self.cls_token_idx).nonzero()[0].item()
            obj_idx = (input_ids[0] == self.obj_token_idx).nonzero()[0].item()
            seg_idx = (input_ids[0] == self.seg_token_idx).nonzero()[0].item()

            cls_vec = hs[0, cls_idx + image_offset]
            obj_vec = hs[0, obj_idx + image_offset]
            seg_vec_raw = hs[0, seg_idx + image_offset]

            # ── Classification ──
            cls_logits = self.model.cls_head[0](cls_vec)  # [3]
            predicted_class = torch.argmax(cls_logits).item()
            cls_probs = torch.softmax(cls_logits, dim=-1)

            cls_label_map = {0: "real", 1: "fully synthetic", 2: "tampered"}
            cls_info = {
                "predicted_class": predicted_class,
                "label": cls_label_map.get(predicted_class, "unknown"),
                "probabilities": {
                    cls_label_map[k]: cls_probs[k].item()
                    for k in range(cls_probs.size(0))
                    if k in cls_label_map
                },
            }

            # Decide whether this sample needs OBJ/seg
            compute_seg_obj = (cls_label == 2) if cls_label is not None else (predicted_class == 2)

            # ── Non-tampered: early return (no generation, no SAM) ──
            if not compute_seg_obj:
                return input_ids, [], None, cls_info

            # ── Stage 2 (tampered only): OBJ head + segmentation ──
            obj_logits = self.model.obj_head(obj_vec)  # [K]
            obj_preds = torch.sigmoid(obj_logits)

            seg_emb = self.model.seg_proj(seg_vec_raw)  # [out_dim]
            mode = getattr(self, "seg_prompt_mode", "fuse")

            if mode == "seg_only":
                # SAM prompt uses only seg_emb; text generation is optional
                if generate_text:
                    output_ids = self.generate(
                        images=images_clip,
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                    )
                else:
                    output_ids = input_ids  # no generated text
                fused = seg_emb
            else:
                # fuse / text_only: need generated text
                output_ids = self.generate(
                    images=images_clip,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                )

                # Second forward pass on full output_ids to get text hidden states
                self.train()
                fwd_output2 = super(PIXARForCausalLM, self).forward(
                    images=images_clip,
                    input_ids=output_ids,
                    output_hidden_states=True,
                )
                if not was_training:
                    self.eval()

                hs2 = fwd_output2.hidden_states[-1]
                T_hidden2 = hs2.shape[1]
                image_offset2 = T_hidden2 - output_ids.shape[1]

                # Recompute seg_idx in output_ids (same position, but offset may differ)
                seg_idx2 = (output_ids[0] == self.seg_token_idx).nonzero()[0].item()
                desc_start = seg_idx2 + image_offset2 + 1
                desc_end = T_hidden2
                desc_h = hs2[0, desc_start:desc_end, :]

                if desc_h.shape[0] > 0:
                    text_vec = desc_h.mean(dim=0)
                else:
                    text_vec = torch.zeros(H_dim, device=hs.device, dtype=hs.dtype)
                text_emb = self.model.text_proj(text_vec)  # [out_dim]

                if mode == "text_only":
                    fused = text_emb
                else:  # fuse
                    gate = torch.sigmoid(self.model.gate_mlp(torch.cat([seg_emb, text_emb], dim=-1)))
                    fused = gate * seg_emb + (1 - gate) * text_emb

            # ── SAM decoder ──
            image_embeddings = self.get_visual_embs(images)
            text_embeds = fused.unsqueeze(0).unsqueeze(1)  # [1, 1, out_dim]
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=text_embeds,
            )
            sparse_embeddings = sparse_embeddings.to(fused.dtype)
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[0].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[0],
                original_size=original_size_list[0],
            )

            return output_ids, [pred_mask[:, 0]], obj_preds, cls_info
