# Evaluation Guide

Two evaluation pipelines are available:

1. **Multi-GPU parallel evaluation** (`test_parallel.py`) — classification, segmentation, and object recognition metrics
2. **Text quality evaluation** (`text_eval/compute_css.py`) — Cosine Semantic Similarity (CSS) between generated descriptions and ground truth

---

## Part 1: Multi-GPU Parallel Evaluation (`test_parallel.py`)

### Prerequisites

| Item | Description |
|---|---|
| Merged model checkpoint | A fully merged (base + LoRA) model in HuggingFace format. Use `merge_lora_weights_and_save_hf_model.py` to produce it. |
| Dataset directory | Test dataset root containing a `validation/` split. |
| SAM ViT-H weights | `sam_vit_h_4b8939.pth` checkpoint. |

### How it works

The script splits the test set into N equal chunks (one per GPU), evaluates them in parallel using spawned subprocesses, then merges all intermediate counts and computes the final metrics — identical to running on the full set sequentially.

### Command

```bash
python test_parallel.py \
  --version /path/to/merged_model \
  --dataset_dir /path/to/test/dataset \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --gpus 0,1,2,3 \
  --output_dir ./evaluation/logs/my_experiment \
  --seg_prompt_mode fuse \
  --precision bf16 \
  --split validation \
  --obj_threshold 0.5 \
  --max_new_tokens 128 \
  --use_mm_start_end \
  --train_mask_decoder \
  --save_generated_text \
  2>&1 | tee ./evaluation/logs/my_experiment/test.log
```

Or use the provided template script and edit the variables at the top:

```bash
bash evaluation/evaluation_PIXAR-7B_ours_seg-only_parallel.sh
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--version` | required | Path to the merged model directory |
| `--dataset_dir` | required | Root directory of the test dataset |
| `--vision_pretrained` | required | Path to SAM ViT-H weights |
| `--gpus` | required | Comma-separated GPU IDs to use, e.g. `0,1,2,3` |
| `--output_dir` | `./test_output_parallel` | Directory to save results |
| `--seg_prompt_mode` | `fuse` | Segmentation prompt mode: `seg_only` / `fuse` / `text_only` |
| `--precision` | `fp16` | Model precision: `fp16` / `bf16` / `fp32` |
| `--split` | `validation` | Dataset split to evaluate |
| `--obj_threshold` | `0.5` | Sigmoid threshold for object prediction |
| `--max_new_tokens` | `128` | Max tokens for text generation |
| `--save_generated_text` | off | Save generated and GT texts to `generated_texts.json` (required for CSS eval) |
| `--load_in_8bit` | off | Load model in 8-bit quantization (useful when VRAM is limited) |
| `--generate_text_in_seg_only` | off | Also generate text tokens in `seg_only` mode |

### Output files

```
output_dir/
├── raw_chunk_0.json          # intermediate counts from GPU 0
├── raw_chunk_1.json          # intermediate counts from GPU 1
├── ...
├── metrics.json              # final merged metrics
├── generated_texts.json      # generated + GT texts (if --save_generated_text)
└── test.log                  # full stdout log
```

### Metrics reported

**Classification**
- Overall accuracy
- Per-class accuracy, precision, recall, F1 (Real / Full Synthetic / Tampered)
- Confusion matrix

**Segmentation** (tampered samples only)
- gIoU, cIoU
- Pixel-level precision, recall, F1
- Pixel ROC-AUC

**Object recognition** (tampered samples only)
- Micro / Macro precision, recall, F1
- Subset accuracy (exact match)
- Top-1 / Top-5 accuracy

---

## Part 2: Text Quality Evaluation (`text_eval/compute_css.py`)

### Prerequisites

A `generated_texts.json` file produced by `test_parallel.py` with `--save_generated_text`. This file contains generated descriptions and ground truth texts for all samples.

The script only evaluates samples where `ground_truth_label == 2` (tampered images).

### Command

Run from the `text_eval/` directory:

```bash
cd evaluation/text_eval

# Print CSS score to stdout only
python compute_css.py \
    --json_path ../logs/my_experiment/generated_texts.json

# Also save per-sample scores to a JSON file
python compute_css.py \
    --json_path ../logs/my_experiment/generated_texts.json \
    --output_path ./logs/my_experiment/css_scores.json
```

Or use the template script:

```bash
cd evaluation/text_eval
# Edit version and gpu at the top of the file, then:
bash evaluation_text.sh
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--json_path` | required | Path to `generated_texts.json` from `test_parallel.py` |
| `--model_name` | `all-MiniLM-L6-v2` | HuggingFace model or local path for sentence embedding |
| `--batch_size` | `512` | Encoding batch size |
| `--output_path` | None | If set, saves per-sample CSS scores to this JSON file |

### Output

```
============================================================
  ground_truth_label=2 samples        : 1000
  Empty generated_text (CSS→0)        : 12 (1.20%)
  Non-empty generated_text            : 988
  Mean CSS  [all, empty→0.0]          : 0.6842
  Mean CSS  [non-empty only]           : 0.6924
  Std  CSS  [non-empty only]           : 0.1103
============================================================
```

- **Mean CSS [all]** — primary metric; empty generated texts contribute a score of 0.0
- **Mean CSS [non-empty only]** — CSS averaged over samples that produced non-empty text
- Per-sample scores (if `--output_path` set) are saved as a JSON list, each entry containing `image_path`, `generated_text`, `gt_text_description`, `css_score`, and `is_empty`

### Two-step workflow

```
test_parallel.py  --save_generated_text
        |
        v
evaluation/logs/<experiment>/generated_texts.json
        |
        v
text_eval/compute_css.py  --json_path ...
        |
        v
text_eval/logs/<experiment>/css_scores.json
```
