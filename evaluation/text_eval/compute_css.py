"""
Compute Cosine Semantic Similarity (CSS) between generated_text and gt_text_description
for samples where ground_truth_label == 2.

Uses sentence-transformers/all-MiniLM-L6-v2 via HuggingFace transformers (no sentence_transformers library needed).

Usage:
    python evaluation/compute_css.py \
        --json_path evaluation/logs/finetune_PIXAR-7B_ours_fuse/generated_texts_full.json

    # Save per-sample scores:
    python evaluation/compute_css.py \
        --json_path evaluation/logs/finetune_PIXAR-7B_ours_fuse/generated_texts_full.json \
        --output_path evaluation/logs/finetune_PIXAR-7B_ours_fuse/css_scores.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Suppress FutureWarning from pynvml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Local cache path for all-MiniLM-L6-v2
_LOCAL_MODEL_PATH = (
    "/data/drstrange/DATA/.cache/huggingface/hub/"
    "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
    "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
_DEFAULT_MODEL = _LOCAL_MODEL_PATH if os.path.isdir(_LOCAL_MODEL_PATH) else "sentence-transformers/all-MiniLM-L6-v2"


def mean_pooling(model_output, attention_mask):
    """Mean pool token embeddings, ignoring padding tokens."""
    token_embeddings = model_output.last_hidden_state  # (B, T, D)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)


def encode_texts(texts, tokenizer, model, batch_size, device):
    """Encode a list of texts into L2-normalized embeddings."""
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        embeddings = mean_pooling(output, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
        if (start // batch_size) % 20 == 0:
            print(f"  Encoded {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return torch.cat(all_embeddings, dim=0)  # (N, D)


def clean_text(text: str) -> str:
    return text.replace("</s>", "").strip()


def compute_css(json_path: str, model_name: str, batch_size: int, output_path: str | None):
    print(f"[INFO] Loading data from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    label2_samples = [d for d in data if d["ground_truth_label"] == 2]
    total_label2 = len(label2_samples)
    print(f"[INFO] Total samples: {len(data)} | ground_truth_label=2: {total_label2}")

    if total_label2 == 0:
        print("[WARN] No samples with ground_truth_label=2 found.")
        return

    generated_texts = [clean_text(d["generated_text"]) for d in label2_samples]
    gt_texts = [clean_text(d["gt_text_description"]) for d in label2_samples]

    empty_mask = [g == "" for g in generated_texts]
    n_empty = sum(empty_mask)
    non_empty_indices = [i for i, e in enumerate(empty_mask) if not e]
    print(f"[INFO] Empty generated_text: {n_empty} / {total_label2} ({100*n_empty/total_label2:.2f}%)")
    print(f"[INFO] Non-empty generated_text: {len(non_empty_indices)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Encode gt texts
    print("[INFO] Encoding gt_text_description ...")
    gt_embeddings = encode_texts(gt_texts, tokenizer, model, batch_size, device)  # (N, D)

    # Encode non-empty generated texts
    embed_dim = gt_embeddings.shape[1]
    gen_embeddings = torch.zeros(total_label2, embed_dim)  # zeros for empty (CSS=0)

    if non_empty_indices:
        print("[INFO] Encoding generated_text (non-empty) ...")
        ne_texts = [generated_texts[i] for i in non_empty_indices]
        ne_embeddings = encode_texts(ne_texts, tokenizer, model, batch_size, device)
        for out_idx, emb in zip(non_empty_indices, ne_embeddings):
            gen_embeddings[out_idx] = emb

    # Cosine similarity: embeddings are L2-normalized, so dot product = cosine sim
    css_scores = (gen_embeddings * gt_embeddings).sum(dim=1).numpy()

    # --- Report ---
    mean_css_all = float(np.mean(css_scores))
    if non_empty_indices:
        ne_scores = css_scores[non_empty_indices]
        mean_css_nonempty = float(np.mean(ne_scores))
        std_css_nonempty = float(np.std(ne_scores))
    else:
        mean_css_nonempty = float("nan")
        std_css_nonempty = float("nan")

    print("\n" + "="*60)
    print(f"  ground_truth_label=2 samples        : {total_label2}")
    print(f"  Empty generated_text (CSS→0)        : {n_empty} ({100*n_empty/total_label2:.2f}%)")
    print(f"  Non-empty generated_text            : {len(non_empty_indices)}")
    print(f"  Mean CSS  [all, empty→0.0]          : {mean_css_all:.4f}")
    print(f"  Mean CSS  [non-empty only]           : {mean_css_nonempty:.4f}")
    print(f"  Std  CSS  [non-empty only]           : {std_css_nonempty:.4f}")
    print("="*60 + "\n")

    if output_path:
        results = []
        for i, sample in enumerate(label2_samples):
            results.append({
                "image_path": sample["image_path"],
                "generated_text": sample["generated_text"],
                "gt_text_description": sample["gt_text_description"],
                "css_score": float(css_scores[i]),
                "is_empty": empty_mask[i],
            })
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Per-sample CSS scores saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute Cosine Semantic Similarity for generated texts.")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to generated_texts_full.json")
    parser.add_argument("--model_name", type=str, default=_DEFAULT_MODEL,
                        help="HuggingFace model name or local path (default: all-MiniLM-L6-v2 local cache)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for encoding (default: 512)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save per-sample CSS scores as JSON")
    args = parser.parse_args()

    compute_css(
        json_path=args.json_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
