"""
PIXAR Interactive Chat Interface

This script provides an interactive command-line interface for testing the PIXAR model.
It allows you to:
1. Input custom prompts
2. Load images interactively
3. Generate text descriptions
4. Visualize segmentation masks

Usage:
    python chat.py --version /path/to/checkpoint --precision bf16

Example:
    python chat.py \
        --version ./runs/pixar_final_v1/ckpt_model \
        --precision fp16 \
        --vision-tower openai/clip-vit-large-patch14
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.PIXAR import PIXARForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="PIXAR Interactive Chat")
    parser.add_argument("--version",
                        default="liuhaotian/llava-llama-2-13b-chat-lightning-preview",
                        help="Path to model checkpoint or HuggingFace model name")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str,
                        help="Directory to save visualization outputs")
    parser.add_argument("--precision", default="bf16", type=str,
                        choices=["fp32", "bf16", "fp16"],
                        help="Precision for inference")
    parser.add_argument("--image_size", default=1024, type=int,
                        help="Image size for SAM")
    parser.add_argument("--model_max_length", default=512, type=int,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", default=8, type=int,
                        help="LoRA rank (if applicable)")
    parser.add_argument("--vision-tower",
                        default="openai/clip-vit-large-patch14",
                        type=str,
                        help="Vision encoder model")
    parser.add_argument("--local-rank", default=0, type=int,
                        help="Local rank for distributed training")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="Load model in 4-bit quantization")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True,
                        help="Use image start/end tokens")
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"],
                        help="Conversation template type")
    parser.add_argument("--max_new_tokens", default=512, type=int,
                        help="Maximum number of tokens to generate")

    # NEW: Object classification arguments
    parser.add_argument("--num_obj_classes", type=int, default=81,
                        help="Number of object categories for <OBJ> token")
    parser.add_argument("--seg_prompt_mode", default="seg_only", type=str,
                        choices=["seg_only", "fuse", "text_only"],
                        help="Segmentation prompt mode")
    parser.add_argument("--generate_text_in_seg_only", action="store_true", default=False,
                        help="In seg_only mode, also generate text description")

    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("🤖 PIXAR Interactive Chat Interface")
    print("="*70)
    print("\nToken Sequence:")
    print("  [CLS] → Classification (real/synthetic/tampered)")
    print("  [OBJ] → Object recognition (81 classes)")
    print("  [SEG] → Segmentation mask generation")
    print("  [END] → Sequence termination")
    print("\nCommands:")
    print("  Type 'quit' or 'exit' to quit")
    print("  Press Ctrl+C to interrupt")
    print("="*70 + "\n")


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    print_banner()

    # ===== Initialize Tokenizer with ALL special tokens =====
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Special tokens are already saved in the tokenizer, just look up their indices
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]
    args.end_token_idx = tokenizer("[END]", add_special_tokens=False).input_ids[0]

    print(f"✅ Tokenizer loaded. Vocabulary size: {len(tokenizer)}")
    print(f"   [CLS] = {args.cls_token_idx}")
    print(f"   [SEG] = {args.seg_token_idx}")
    print(f"   [OBJ] = {args.obj_token_idx}")
    print(f"   [END] = {args.end_token_idx}")

    # ===== Set precision =====
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # ===== Quantization config =====
    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    # ===== Load Model =====
    print(f"\n🔧 Loading model from: {args.version}")
    print(f"   Precision: {args.precision}")

    model = PIXARForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        cls_token_idx=args.cls_token_idx,
        obj_token_idx=args.obj_token_idx,
        num_obj_classes=args.num_obj_classes,
        seg_prompt_mode=args.seg_prompt_mode,
        **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU (slow!)")

    # Initialize vision modules
    try:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
        print("✅ Vision modules initialized")
    except AttributeError:
        print("⚠️  Vision tower initialization skipped")

    # Set precision
    if args.precision == "bf16":
        model = model.bfloat16()
        if torch.cuda.is_available():
            model = model.cuda()
    elif args.precision == "fp16":
        model = model.half()
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        model = model.float()
        if torch.cuda.is_available():
            model = model.cuda()

    # Initialize processors
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    print("✅ Model ready for inference\n")

    # ===== Interactive Loop =====
    try:
        while True:
            print("\n" + "-"*70)

            # Get image path
            image_path = input("📁 Image path (or 'quit' to exit): ").strip()
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue

            # Get prompt (optional)
            print("\n💬 Prompt (press Enter for default):")
            user_prompt = input("   ").strip()

            if not user_prompt:
                user_prompt = (
                    "Can you identify whether this image is real, fully synthetic, or tampered? "
                    "If it is tampered, please (1) classify which object was modified and "
                    "(2) output a mask for the modified regions."
                )
                print(f"   Using default: {user_prompt[:80]}...")

            # Prepare conversation
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []

            prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "[CLS] [OBJ] [SEG] ")
            prompt = conv.get_prompt()

            # Load and preprocess image
            print("\n🖼️  Loading image...")
            image_np = cv2.imread(image_path)
            if image_np is None:
                print(f"❌ Failed to load image: {image_path}")
                continue

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]
            print(f"   Image size: {image_np.shape[1]}x{image_np.shape[0]}")

            # Prepare CLIP image
            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda() if torch.cuda.is_available() else
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0].unsqueeze(0)
            )
            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()

            # Prepare SAM image
            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
            )
            if torch.cuda.is_available():
                image = image.cuda()
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()

            # Tokenize input
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Generate output
            print("🤖 Generating response...")
            with torch.no_grad():
                output_ids, pred_masks, obj_preds, cls_info = model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=args.max_new_tokens,
                    tokenizer=tokenizer,
                    generate_text=args.generate_text_in_seg_only,
                )

            # Decode only the newly generated tokens (skip the input portion)
            input_token_len = input_ids.shape[1]
            new_tokens = output_ids[0][input_token_len:]
            new_tokens = new_tokens[new_tokens != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text_output = text_output.replace("\n", " ").replace("  ", " ").strip()

            # Determine predicted class
            predicted_class = cls_info["predicted_class"]
            label = cls_info["label"]

            # Print classification result
            print("\n" + "="*70)
            icons = {"real": "✅ REAL", "fully synthetic": "🤖 FULLY SYNTHETIC", "tampered": "⚠️  TAMPERED"}
            print(f"   Classification: {icons.get(label, label.upper())}")
            for name, prob in cls_info["probabilities"].items():
                print(f"     - {name}: {prob:.4f}")
            print("="*70)

            if predicted_class == 0:
                # Real
                print("\n📝 Result: This image is real.")

            elif predicted_class == 1:
                # Fully synthetic
                print("\n📝 Result: This image is fully synthetic.")

            else:
                # Tampered — display generated text, seg mask, and obj predictions
                print("\n📝 Generated Description:")
                print(f"   {text_output}")

                # Assert tampered must have segmentation mask and object predictions
                assert len(pred_masks) > 0, \
                    "Tampered prediction but no segmentation mask was produced!"
                assert obj_preds is not None and obj_preds.numel() > 0, \
                    "Tampered prediction but no object classification was produced!"

                # Object classification
                OBJ_CLASS_NAMES = [
                    "person", "bicycle", "car", "motorcycle", "airplane",
                    "bus", "train", "truck", "boat", "traffic light",
                    "fire hydrant", "stop sign", "parking meter", "bench",
                    "bird", "cat", "dog", "horse", "sheep", "cow",
                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                    "sports ball", "kite", "baseball bat", "baseball glove",
                    "skateboard", "surfboard", "tennis racket", "bottle",
                    "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                    "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake",
                    "chair", "couch", "potted plant", "bed", "dining table",
                    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                    "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush", "background"
                ]
                detected = (obj_preds > 0.5).nonzero(as_tuple=True)[0]
                if len(detected) > 0:
                    names = [OBJ_CLASS_NAMES[idx] for idx in detected if idx < len(OBJ_CLASS_NAMES)]
                    print(f"   Modified objects: {', '.join(names)}")
                else:
                    print("   Modified objects: (none above threshold)")

                # Save masks
                print(f"\n💾 Saving {len(pred_masks)} mask(s)...")
                for i, pred_mask in enumerate(pred_masks):
                    if pred_mask.shape[0] == 0:
                        continue

                    pred_mask = pred_mask.detach().cpu().numpy()[0]
                    pred_mask = pred_mask > 0

                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    mask_path = os.path.join(
                        args.vis_save_path,
                        f"{base_name}_mask_{i}.jpg"
                    )
                    cv2.imwrite(mask_path, pred_mask.astype(np.uint8) * 255)
                    print(f"   ✅ Mask saved: {mask_path}")

                    overlay_path = os.path.join(
                        args.vis_save_path,
                        f"{base_name}_overlay_{i}.jpg"
                    )
                    vis_img = image_np.copy()
                    vis_img[pred_mask] = (
                        image_np * 0.5
                        + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                    )[pred_mask]
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(overlay_path, vis_img)
                    print(f"   ✅ Overlay saved: {overlay_path}")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1:])
