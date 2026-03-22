# Dataset Construction

This directory contains all scripts needed to build the **PIXAR dataset** from raw image pairs at a chosen pixel-difference threshold τ. The pipeline computes per-pixel difference maps between original and AI-generated images, thresholds them to produce soft masks $M_\tau$, and assembles the final training/test splits with optional text descriptions.

---

## Directory Contents

| File / Folder | Description |
|---|---|
| `2_construct_dataset_text.py` | Core processing script — computes diff maps, writes masks and metadata (class labels + text descriptions from CSV) |
| `generate_v2.sh` | Batch runner for the **training set** (mask-only labels, no text) |
| `generate_v2-text.sh` | Batch runner for the **training set** (masks + text descriptions) |
| `generate_v2-text-val.sh` | Batch runner for the **test set** — single generative source, configurable via the `TYPE` variable |
| `generate_v2-text-val_all.sh` | Convenience wrapper that runs all per-source test set scripts in sequence |
| `rebuild_descriptions_csv.py` / `.sh` | Rebuilds the descriptions CSV from existing metadata if needed |
| `verify_metadata.py` / `.sh` | Verifies the integrity of generated metadata JSONs |
| `logs/` | Runtime logs (one file per run, named by timestamp) |

---

## Quick Start

### Step 0 — Download description CSV files

Download the description CSV files from [Google Drive](https://drive.google.com/drive/folders/1ESIQziludQuW_ECGh_xWCP8IOKU-mKe1?usp=drive_link) and place them in a local directory. These files provide the natural language descriptions used during dataset construction.

### Step 1 — Build the training set

Open `generate_v2-text.sh` (with text descriptions) or `generate_v2.sh` (mask-only) and edit the config block at the top:

```bash
DATASET_DIR="/path/to/raw_outputs"           # output of download-data/download.sh
OUT_DIR="/path/to/output/train/ours"         # where to write the processed dataset
TAOS=(0.05)                                  # one or more τ values, e.g. (0.01 0.05 0.1)

# only for generate_v2-text.sh:
DESCRIPTIONS_CSV="/path/to/descriptions_train.csv"
```

Then run from this directory:

```bash
cd utils_preprocess/construct_dataset

# mask-only labels
bash generate_v2.sh

# labels + text descriptions (recommended)
bash generate_v2-text.sh
```

### Step 2 — Build the test set

Edit `generate_v2-text-val.sh`, setting the `TYPE` variable to the desired generative source:

```bash
TYPE="qwen"   # one of: gemini | gemini3 | gpt | flux2 | qwen | seedream
TAOS=(0.05)
DATASET_DIR="/path/to/raw_outputs"
OUT_DIR="/path/to/output/test/${TYPE}"
DESCRIPTIONS_CSV="/path/to/descriptions.csv"
```

Then run:

```bash
bash generate_v2-text-val.sh
```

To process **all sources at once**, use:

```bash
bash generate_v2-text-val_all.sh
```

---

## Annotation Modes

The core script `2_construct_dataset_text.py` accepts two optional flags that control how per-sample annotations are extracted:

| Flag | When to use | Effect |
|---|---|---|
| `--anno` | Splits with COCO instance annotations (replacement, removal) | Reads COCO JSON to extract object category labels for the tampered region |
| `--bg` | Background-type splits | Treats the entire image as background; no instance-level annotation required |
| *(neither)* | Addition, color, motion, material splits | Derives labels directly from the edit metadata without COCO annotations |

The batch scripts automatically pass the correct flags for each split group.

---

## Training Split Groups

The training set is divided into sub-splits based on the manipulation type:

| Group | Split IDs | Flags |
|---|---|---|
| Replacement (inter-class) | `coco_train_inter_replacement_1/2` | `--anno` |
| Replacement | `coco_train_replacement_1/2` | `--anno` |
| Removal | `coco_train_removal_1` | `--anno --bg` |
| Addition / Color / Motion / Material | `coco_train_{addition,color,motion,material}` | *(none)* |
| Background | `coco_train_background` | `--bg` |

Test splits follow the same structure, prefixed with the generative source name (e.g. `qwen_coco_val_replacement_1`). Note that `gemini3` does not include a motion split.

---

## τ Selection Guide

| τ | Effect |
|---|---|
| 0.01 | Captures micro-edits and subtle pixel changes |
| **0.05** | **Default — balanced sensitivity (recommended)** |
| 0.1 | High-confidence semantic changes only |
| 0.2 | Conservative — only large, obvious edits |

Multiple τ values can be processed in a single run:

```bash
TAOS=(0.5 0.1 0.2)
```

---

## Output Structure

```
OUT_DIR/
├── train/
│   ├── real/               # Original (authentic) images
│   ├── full_synthetic/     # Fully AI-generated images
│   ├── tampered/           # Tampered images (AI-edited)
│   ├── masks/              # Hard binary masks (thresholded at τ)
│   ├── soft_masks/         # Pixel-difference maps M_τ (float, same τ)
│   └── metadata/           # Per-image JSON: {"cls": [...], "text": "..."}
└── validation/
    └── (same structure)
```

Each run writes a timestamped log to `logs/construct_unified_text_<YYYYMMDD_HHMMSS>.log`.