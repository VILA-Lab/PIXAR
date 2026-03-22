#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers.modeling_utils import load_sharded_checkpoint

from model.PIXAR import PIXARForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def parse_args(argv):
    parser = argparse.ArgumentParser(description="merge lora weights and save model with hf format")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)

    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)

    parser.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--weight", required=True, help="LoRA ckpt file or HF-sharded dir")
    parser.add_argument("--save_path", required=True, help="output dir to save merged model")
    return parser.parse_args(argv)


def find_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                x not in name
                for x in [
                    "visual_model",
                    "vision_tower",
                    "mm_projector",
                    "cls_head",
                    "obj_head",
                    "seg_proj",
                    "text_proj",
                    "gate_mlp",
                ]
            )
            and any(x in name for x in lora_target_modules)
        ):
            names.add(name)
    return sorted(list(names))


def main(argv):
    args = parse_args(argv)
    os.makedirs(args.save_path, exist_ok=True)

    # ------ 1) tokenizer：顺序必须与训练一致 ------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # 训练顺序：CLS -> SEG -> OBJ -> END -> (im_start, im_end)
    # 顺序必须与 train_PIXAR.py, test.py, chat.py 保持一致
    tokenizer.add_tokens("[CLS]")
    tokenizer.add_tokens("[SEG]")
    tokenizer.add_tokens("[OBJ]")
    tokenizer.add_tokens("[END]")  # << 新增：序列结束标记
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]
    end_token_idx = tokenizer("[END]", add_special_tokens=False).input_ids[0]

    # ------ 2) 构建模型（与训练侧一致）------
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": seg_token_idx,
        "cls_token_idx": cls_token_idx,
        "obj_token_idx": obj_token_idx,
        # end_token_idx 在当前模型中可能不直接使用，但保持一致性
        "vision_tower": args.vision_tower,
        # 若训练时显式设置了 num_obj_classes，这里也可传；默认 81
        # "num_obj_classes": 81,
    }

    model = PIXARForCausalLM.from_pretrained(
        args.version, torch_dtype=dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=dtype)
    model.get_model().initialize_pixar_modules(model.get_model().config)

    # ------ 3) 挂 LoRA 外壳（与训练相同的 target_modules）------
    if args.lora_r > 0:
        targets = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=targets, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # 关键：在加载权重前，先把 embedding 大小对齐到当前 tokenizer（已含 [OBJ]）
    model.resize_token_embeddings(len(tokenizer))

    # ------ 4) 加载 LoRA 权重（此时 vocab 尺寸应与 ckpt 一致，避免 size mismatch）------
    if os.path.isdir(args.weight):
        # HF 分片目录（包含 pytorch_model.bin.index.json）
        load_sharded_checkpoint(model, args.weight, strict=True)
    else:
        state_dict = torch.load(args.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)

    # ------ 5) 合并 LoRA 并保存 ------
    model = model.merge_and_unload()

    # 通常不保存 vision_tower，减小体积；按需可去掉这个 if
    to_save = {k: v for k, v in model.state_dict().items() if "vision_tower" not in k}
    model.save_pretrained(args.save_path, state_dict=to_save)
    tokenizer.save_pretrained(args.save_path)
    print(f"✅ Merged model saved to: {args.save_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
