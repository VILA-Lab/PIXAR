
# Fine-tune PIXAR-7B with DeepSpeed on a single GPU (multi-task: classification / segmentation / OBJ).
# Usage notes:
# 1) GPU selection:
#    --include localhost:2 means only GPU 2 is used; change to localhost:0/1/3 etc. for other GPUs.
# 2) Port setting:
#    --master_port must not be occupied on the current machine; change the port number if there is a conflict.
# 3) Model and data:
#    --version           path to PIXAR initial weights directory (HuggingFace format)
#    --dataset_dir       root directory of training data (should contain train/validation splits or equivalent structure)
#    --val_dataset       validation set path (used for periodic evaluation and saving the best checkpoint)
#    --vision_pretrained SAM ViT-H weights path (used by the segmentation module)
# 4) Training configuration:
#    --batch_size        per-GPU micro-batch size (total batch = batch_size x grad_accumulation_steps)
#    --epochs            number of training epochs
#    --steps_per_epoch   training steps per epoch (depends on dataset size / sampling strategy)
#    --lr                learning rate (1e-4 or smaller recommended for bf16)
#    --dice_loss_weight  weight of Dice loss in the segmentation task
#    --precision         compute precision; bf16 requires hardware support
# 5) Logging and output:
#    --exp_name          experiment name (used to distinguish runs under the runs directory)
#    --log_base_dir      root directory for saving TensorBoard logs and checkpoints

################################################################################
# Key parameters — edit here
################################################################################
BASE_DIR="path/to/PIXAR"

GPU="localhost:"
PORT=12532
VERSION="${BASE_DIR}/ck/SIDA-7B"
DATASET_DIR="path/to/ours_0.05"
VAL_DATASET="path/to/ours_0.05/validation"
VISION_PRETRAINED="${BASE_DIR}/ck/sam_vit_h_4b8939.pth"
LOG_BASE_DIR="${BASE_DIR}/runs"
EXP_NAME="PIXAR-7B"

BATCH_SIZE=2
EPOCHS=20
STEPS_PER_EPOCH=1000
LR=0.0001
PRECISION="bf16"

DICE_LOSS_WEIGHT=1.0
OBJ_LOSS_WEIGHT=0.5
TEXT_LOSS_WEIGHT=3.0
SEG_PROMPT_MODE="seg_only"
MASK_TYPE="ours"          # "ours" -> gt_soft_mask, "others" -> gt_mask
################################################################################

mkdir -p ./finetune/logs

deepspeed --include ${GPU} --master_port=${PORT} train_PIXAR.py \
  --version="${VERSION}" \
  --dataset_dir="${DATASET_DIR}" \
  --vision_pretrained="${VISION_PRETRAINED}" \
  --val_dataset="${VAL_DATASET}" \
  --batch_size=${BATCH_SIZE} \
  --exp_name="${EXP_NAME}" \
  --epochs=${EPOCHS} \
  --dice_loss_weight ${DICE_LOSS_WEIGHT} \
  --obj_loss_weight ${OBJ_LOSS_WEIGHT} \
  --mask_type "${MASK_TYPE}" \
  --seg_prompt_mode "${SEG_PROMPT_MODE}" \
  --steps_per_epoch=${STEPS_PER_EPOCH} \
  --precision="${PRECISION}" \
  --lr=${LR} \
  --text_loss_weight ${TEXT_LOSS_WEIGHT} \
  --no_eval \
  --log_base_dir="${LOG_BASE_DIR}" > ./finetune/logs/${EXP_NAME}.log 2>&1 &
