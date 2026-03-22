CUDA_VISIBLE_DEVICES="0" python merge_lora_weights_and_save_hf_model.py \
  --version="/data/ironman/jiacheng/final_Omni_Data/ck/LISA-7B" \
  --weight="/data/ironman/jiacheng/final_Omni_Data/runs/finetune_LISA-7B/pytorch_model.bin" \
  --save_path="/data/ironman/jiacheng/final_Omni_Data/ck/finetune_LISA-7B_20ep"