import glob
import os
import random
import json  # <-- NEW

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, cls_token_idx=None, obj_token_idx=None, seg_token_idx=None,
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    soft_masks_list = []
    label_list = []
    cls_labels_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    has_text_description = []
    obj_labels_list = []   # <-- NEW
    
    # Process batch items
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        soft_masks,      # 新增
        label,
        cls_labels,
        resize,
        questions, 
        sampled_classes,
        inference,
        has_text,
        obj_label_idx
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        soft_masks_list.append(soft_masks.float())  # 新增
        label_list.append(label)
        cls_labels_list.append(torch.tensor(cls_labels))
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        has_text_description.append(has_text)
        # All samples now have obj_label_vec (zeros for non-tampered)
        if torch.is_tensor(obj_label_idx):
            obj_labels_list.append(obj_label_idx.to(torch.float32))
        else:
            obj_labels_list.append(torch.tensor(obj_label_idx, dtype=torch.float32))

    # Handle image tokens
    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # Pre-calculate original lengths before padding
    original_input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    original_lengths = [len(ids) for ids in original_input_ids]

    # === Sanity check: [OBJ] token count should match obj_labels count ===
    if obj_token_idx is not None:
        obj_token_count = sum((ids == obj_token_idx).sum().item() for ids in original_input_ids)
        if obj_token_count != len(obj_labels_list):
            raise RuntimeError(
                f"[OBJ] mismatch: tokens={obj_token_count}, labels={len(obj_labels_list)}. "
                "Every sample must have exactly one [OBJ] token."
            )

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        original_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Process targets using original lengths
    targets = []
    for i, conversation in enumerate(conversation_list):
        if has_text_description[i]:
            target = input_ids[i].clone()
        else:
            target = torch.full_like(input_ids[i], IGNORE_INDEX)
        targets.append(target)

    targets = torch.stack(targets)
    conv = conversation_lib.default_conversation.copy()
    
    # Set separator based on conversation type
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "
    
    # Process each conversation using original lengths
    for idx, (conversation, target, orig_len) in enumerate(zip(conversation_list, targets, original_lengths)):
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(rounds):
            if rou == "":
                break
                
            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"Warning: Unexpected format in conversation {idx}")
                continue
                
            parts[0] += sep
            
            # Calculate lengths
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        
        # Use original length for verification
        total_len = orig_len
        
        if cur_len != total_len:
            print(f"Length mismatch in conversation {idx}:")
            print(f"cur_len: {cur_len}, total_len: {total_len}")
            print(f"conversation: {conversation}")
            
        # Keep the assertion as a safety check
        assert cur_len == total_len, f"Length mismatch: cur_len={cur_len}, total_len={total_len}"
        
        target[cur_len:] = IGNORE_INDEX

    # Force IGNORE_INDEX on structure tokens — never supervised by LM loss
    if cls_token_idx is not None:
        targets[input_ids == cls_token_idx] = IGNORE_INDEX
    if obj_token_idx is not None:
        targets[input_ids == obj_token_idx] = IGNORE_INDEX
    if seg_token_idx is not None:
        targets[input_ids == seg_token_idx] = IGNORE_INDEX

    # Handle truncation for non-inference cases
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "cls_labels": torch.stack(cls_labels_list).view(-1),
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "soft_masks_list": soft_masks_list, 
        "cls_labels_list": cls_labels_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        # multi-hot: [B, K] — all samples have obj_labels (zeros for non-tampered)
        "obj_labels": torch.stack(obj_labels_list, dim=0),
    }

class CustomDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,  # Root directory containing real/full_synthetic/tampered
        tokenizer,
        vision_tower,
        split="train",
        precision: str = "fp32",
        image_size: int = 224,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.split = split
        # Image processing
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        # Set up paths
        split_dir = os.path.join(base_image_dir, split)
        required_dirs = ["real", "full_synthetic", "tampered"]
        for dir_name in required_dirs:
            dir_path = os.path.join(split_dir, dir_name)
            if not os.path.exists(dir_path):
                raise ValueError(f"Required directory {dir_path} does not exist!")

        # Load images and verify
        self.images = []
        self.cls_labels = []
        self.invalid_samples = []  # Track problematic samples

        # === NEW: 类别映射（举例：COCO-80；你也可以从外部json/csv加载）===
        self.class_to_idx = {
            "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
            "bus": 5, "train": 6, "truck": 7, "boat": 8,
            "traffic light": 9, "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
            "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
            "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24, "umbrella": 25,
            "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29, "skis": 30, "snowboard": 31,
            "sports ball": 32, "kite": 33, "baseball bat": 34, "baseball glove": 35,
            "skateboard": 36, "surfboard": 37, "tennis racket": 38, "bottle": 39,
            "wine glass": 40, "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
            "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50,
            "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55,
            "chair": 56, "couch": 57, "potted plant": 58, "bed": 59, "dining table": 60,
            "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64, "remote": 65, "keyboard": 66,
            "cell phone": 67, "microwave": 68, "oven": 69, "toaster": 70, "sink": 71,
            "refrigerator": 72, "book": 73, "clock": 74, "vase": 75, "scissors": 76,
            "teddy bear": 77, "hair drier": 78, "toothbrush": 79, "background": 80
        }
        self.num_obj_classes = len(self.class_to_idx)  # 便于构造 multi-hot

        # Load images and verify counts
        real_images = glob.glob(os.path.join(split_dir, "real", "*.jpg"))
        real_images += glob.glob(os.path.join(split_dir, "real", "*.png"))
        full_syn_images = glob.glob(os.path.join(split_dir, "full_synthetic", "*.png"))
        tampered_images = glob.glob(os.path.join(split_dir, "tampered", "*.png"))

        # Verify tampered images have corresponding masks
        valid_tampered_images = []
        for img_path in tampered_images:
            # Extract the base filename without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Construct the mask path (assuming the mask filename appends '_mask' to the base name)
            mask_name = f"{base_name}_mask.png"
            mask_path = os.path.join(split_dir, "masks", mask_name)
            # Check if the mask exists
            if os.path.exists(mask_path):
                valid_tampered_images.append(img_path)
            else:
                print(f"Mask not found for: {img_path}")

        # Add only valid images to the dataset
        self.images.extend(real_images)
        self.images.extend(full_syn_images)
        self.images.extend(valid_tampered_images)  # Use valid_tampered_images here

        # Assign labels based on the valid counts
        self.cls_labels.extend([0] * len(real_images))
        self.cls_labels.extend([1] * len(full_syn_images))
        self.cls_labels.extend([2] * len(valid_tampered_images))  # Use valid_tampered_images here

        # Print dataset statistics
        print(f"\nDataset Statistics for {split} split:")
        print(f"Real images: {len(real_images)}")
        print(f"Full synthetic images: {len(full_syn_images)}")
        print(f"Tampered images: {len(valid_tampered_images)} (Valid) / {len(tampered_images)} (Total)")
        if self.invalid_samples:
            print(f"Warning: Found {len(self.invalid_samples)} invalid samples")

    def __len__(self):
        return len(self.images)
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def _generate_response(self, cls_label, text_description=None):
        """Generate response with unified [CLS] [OBJ] [SEG] prefix for ALL samples."""
        if cls_label == 0:
            response = "[CLS] [OBJ] [SEG]"
        elif cls_label == 1:
            response = "[CLS] [OBJ] [SEG]"
        else:  # cls_label == 2 (tampered)
            response = "[CLS] [OBJ] [SEG] The image is tampered."
            if text_description:
                response += f" {text_description}"

        return response
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_name = os.path.basename(image_path)
        cls_labels = self.cls_labels[idx]
        # Load and process image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process for CLIP
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # Process image for model
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # Initialize mask placeholders
        mask = torch.zeros((1, resize[0], resize[1]))
        soft_mask = torch.zeros((1, resize[0], resize[1]))

        obj_label_vec = torch.zeros(self.num_obj_classes, dtype=torch.float32)  # all-zeros for non-tampered
        text_description = None

        # Load mask for tampered
        if cls_labels == 2:
            base_name = os.path.splitext(image_name)[0]
            mask_name = f"{base_name}_mask.png"
            # mask
            mask_path = os.path.join(self.base_image_dir, self.split, "masks", mask_name)
            
            # soft mask
            soft_mask_path = os.path.join(self.base_image_dir, self.split, "soft_masks", mask_name)

            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_img = self.transform.apply_image(mask_img)
                mask_img = mask_img / 255.0
                mask = torch.from_numpy(mask_img).unsqueeze(0)

            if os.path.exists(soft_mask_path):
                soft_mask_img = cv2.imread(soft_mask_path, cv2.IMREAD_GRAYSCALE)
                soft_mask_img = self.transform.apply_image(soft_mask_img)
                soft_mask_img = soft_mask_img / 255.0
                soft_mask = torch.from_numpy(soft_mask_img).unsqueeze(0)
            else:
                # 若soft mask缺失，fallback为hard mask
                print(f"Soft mask not found for: {image_name}, using hard mask as fallback.")
                soft_mask = mask.clone()
            
            # === 读取 tamper 元数据，产出 multi-hot 向量和 text description ===
            meta_path = os.path.join(self.base_image_dir, self.split, "metadata", f"{base_name}_cls.json")
            text_description = None
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                # 现在 cls 是一个 list（可能是类名字符串，也可能是类索引），不再用 [0]
                cls_list = meta.get("cls", [])
                # 读取 text description
                text_description = meta.get("text", None)

                # Build multi-hot vector
                for c in cls_list:
                    if isinstance(c, str):
                        if c in self.class_to_idx:
                            obj_label_vec[self.class_to_idx[c]] = 1.0
                        else:
                            print(f"[WARN] Unknown class '{c}' in metadata for {base_name}")
                    elif isinstance(c, int):
                        if 0 <= c < self.num_obj_classes:
                            obj_label_vec[c] = 1.0
                        else:
                            print(f"[WARN] Class index {c} out of range for {base_name}")
                    else:
                        print(f"[WARN] Unsupported class entry {c} for {base_name}")

            except Exception as e:
                print(f"[WARN] fail to read metadata for {base_name}: {e}")
                text_description = None


        # Generate conversation
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0],
            f"{DEFAULT_IMAGE_TOKEN}\nCan you identify whether this image is real, fully synthetic, or tampered? If it is tampered, please (1) classify which object was modified and (2) output a mask for the modified regions.")

        # Generate response with unified [CLS] [OBJ] [SEG] prefix
        response = self._generate_response(cls_labels, text_description)

        conv.append_message(conv.roles[1], response)
        conversation = conv.get_prompt()

        # Only tampered samples have text after [SEG]
        has_text = (cls_labels == 2)

        labels = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label

        return image_path, image, image_clip, [conversation], mask, soft_mask, labels, cls_labels, resize, None, None, False, has_text, obj_label_vec


    def __len__(self):
        return len(self.images)
