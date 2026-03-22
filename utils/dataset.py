import glob
import os
import random

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
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, stage = 1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(torch.tensor(label)) 
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    device = images_list[0].device  # Assuming images_list tensors are already on the right device
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    if stage == 1:
        labels = torch.stack(label_list).view(-1)
        return{
            "image_paths": image_path_list,
            "images": torch.stack(images_list, dim=0),
            "images_clip": torch.stack(images_clip_list, dim=0),
            "input_ids": input_ids,
            "labels": labels,  # Correctly formatted for classification
            "attention_masks": attention_masks,
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "sampled_classes_list": sampled_classes_list,
            "inference": inferences[0],
            "conversation_list": conversation_list,
        }
    else:
        conv = conversation_lib.default_conversation.copy()
        targets = input_ids.clone()

        if conv_type == "llava_v1":
            sep = conv.sep + conv.roles[1] + ": "
        else:
            sep = "[/INST] "
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if False:
                z = target.clone()
                z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
                if local_rank == 0:
                    print(
                        "conversation: ",
                        conversation,
                        "tokenizer.decode(z): ",
                        tokenizer.decode(z),
                    )

            if cur_len < tokenizer.model_max_length:
                assert cur_len == total_len

        if inferences[0] == False:
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
            "labels": targets,
            "attention_masks": attention_masks,
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "sampled_classes_list": sampled_classes_list,
            "inference": inferences[0],
            "conversation_list": conversation_list,
    }
class CustomDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,  # Root directory containing real/full_synthetic/object_synthetic
        tokenizer,
        vision_tower,
        split="train",
        precision: str = "fp32",
        image_size: int = 224,
        stage: int = 1,  # Add stage parameter
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.stage = stage
        self.split = split
        # Image processing
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # Set up paths
        split_dir = os.path.join(base_image_dir, split)
        required_dirs = ["real", "full_synthetic", "object_part_synthetic"]
        for dir_name in required_dirs:
            dir_path = os.path.join(split_dir, dir_name)
            if not os.path.exists(dir_path):
                raise ValueError(f"Required directory {dir_path} does not exist!")

        # Load images and verify
        self.images = []
        self.labels = []
        self.invalid_samples = []  # Track problematic samples

        # Load images and verify counts
        real_images = glob.glob(os.path.join(split_dir, "real", "*.jpg"))
        full_syn_images = glob.glob(os.path.join(split_dir, "full_synthetic", "*.png"))
        obj_part_syn_images = glob.glob(os.path.join(split_dir, "object_part_synthetic", "*.png"))
        
        # Verify object/part synthetic images have corresponding masks
        valid_obj_part_syn_images = []
        for img_path in obj_part_syn_images:
            # Extract the base filename without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
    
            # Construct the mask path (assuming the mask filename appends '_mask' to the base name)
            mask_name = f"{base_name}_mask.png"
            mask_path = os.path.join(split_dir, "masks", mask_name)
    
            # Check if the mask exists
            if os.path.exists(mask_path):
                valid_obj_part_syn_images.append(img_path)
            else:
                print(f"Mask not found for: {img_path}")

        self.images.extend(real_images)
        self.images.extend(full_syn_images)
        self.images.extend(obj_part_syn_images)
        
        self.labels.extend([0] * len(real_images))
        self.labels.extend([1] * len(full_syn_images))
        self.labels.extend([2] * len(obj_part_syn_images))
         # Print dataset statistics
        print(f"\nDataset Statistics for {split} split:")
        print(f"Real images: {len(real_images)}")
        print(f"Full synthetic images: {len(full_syn_images)}")
        print(f"Object/part synthetic images: {len(valid_obj_part_syn_images)} (Valid) / {len(obj_part_syn_images)} (Total)")

        if self.invalid_samples:
            print(f"Warning: Found {len(self.invalid_samples)} invalid samples")
        # Load text descriptions (for 3k images)
        self.text_descriptions = {}
        if split == "train":
            desc_dir = os.path.join(base_image_dir, "text_label_images", "description")
            if not os.path.exists(desc_dir):
                print(f"Warning: Description directory {desc_dir} does not exist!")
                print(f"Images with descriptions: {len(self.text_descriptions)}")

                # Load descriptions for each type
                for img_type in ["real", "full_synthetic", "object_part_synthetic"]:
                    desc_file = os.path.join(desc_dir, img_type, "descriptions_GPT.txt")
                    if os.path.exists(desc_file):
                        current_img = None
                        current_desc = []
                        with open(desc_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("Image:"):
                                    if current_img and current_desc:
                                        self.text_descriptions[current_img] = "\n".join(current_desc)

                                    # Start new image
                                    current_img = line.split(": ")[1].strip()
                                    current_desc = []
                                elif line.startswith("Description:"):
                                    continue
                                elif line:
                                    current_desc.append(line)
                            if current_img and current_desc:
                                self.text_descriptions[current_img] = "\n".join(current_desc)
                print(f"Loaded {len(self.text_descriptions)} text descriptions")
         # For Stage 2, load mask paths for object/part synthetic images
        self.mask_paths = {}
        if stage == 2:
            for img_path in self.images:
                if "object_part_synthetic" in img_path:
                    mask_path = img_path.replace("object_part_synthetic", "masks")
                    if os.path.exists(mask_path):
                        self.mask_paths[img_path] = mask_path
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
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_name = os.path.basename(image_path)
        label = self.labels[idx]

        # Load and process image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process for CLIP
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # Process image for model
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.stage == 1:
            # Stage 1: Classification
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], 
                f"{DEFAULT_IMAGE_TOKEN}\nCan you identify if this image is real, full synthetic, or object/part synthetic? Please mask the tampered object/part if it is object/part synthetic?")
            conv.append_message(conv.roles[1], "[CLS]")
            conversation = conv.get_prompt()
            conversations = [conversation]

            return image_path, image, image_clip, conversations, torch.zeros(1), label, resize, None, None, False
        else:
            # Stage 2: Segmentation and Text
            # Handle mask
            if self.stage == 2:
                if label == 2:  # object/part synthetic
                    mask_path = self.mask_paths.get(image_path)
                    if mask_path and os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        # Convert to binary (0-1)
                        mask = (mask > 0).astype(np.float32)  # Any non-zero value becomes 1
                        mask = torch.from_numpy(mask)
                    else:
                        mask = torch.ones((image.shape[1], image.shape[2])) * self.ignore_label
                else:  # real or full synthetic
            # Use ignore_label mask - no mask loss will be calculated
                    mask = torch.ones((image.shape[1], image.shape[2])) * self.ignore_label
            # Handle text description
            conv = conversation_lib.default_conversation.copy()

            if self.split == "train":
                has_text = image_name in self.text_descriptions
                if has_text:
                    # Has description -> use it for text loss
                    text = self.text_descriptions[image_name]
                    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{text}")
                else:
                # No description -> use default prompt, no text loss
                    conv.append_message(conv.roles[0], 
                        f"{DEFAULT_IMAGE_TOKEN}\nCan you identify if this image is real, full synthetic, or object/part synthetic? Please mask the tampered object/part if it is object/part synthetic?")
            conv.append_message(conv.roles[1], "[SEG]")
            conversation = conv.get_prompt()
            conversations = [conversation]

            return image_path, image, image_clip, conversations, mask, label, resize, has_text, None, False
