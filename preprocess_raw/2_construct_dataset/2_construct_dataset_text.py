#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import json
import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import csv


def sync_ai_image_size(ai_image_path: str, real_image_path: str) -> bool:
    """
    读取 ai_image 与 real_image；若尺寸不同，则将 ai_image resize 到 real_image 尺寸并覆盖保存。
    返回值：是否发生了 resize（True/False）
    """
    ai = cv2.imread(ai_image_path, cv2.IMREAD_UNCHANGED)
    real = cv2.imread(real_image_path, cv2.IMREAD_UNCHANGED)

    if ai is None:
        raise FileNotFoundError(f"无法读取 ai_image: {ai_image_path}")
    if real is None:
        raise FileNotFoundError(f"无法读取 real_image: {real_image_path}")

    h_ai, w_ai = ai.shape[:2]
    h_real, w_real = real.shape[:2]

    if (h_ai, w_ai) == (h_real, w_real):
        return False

    interp = cv2.INTER_CUBIC if (h_ai < h_real or w_ai < w_real) else cv2.INTER_AREA
    ai_resized = cv2.resize(ai, (w_real, h_real), interpolation=interp)

    ok = cv2.imwrite(ai_image_path, ai_resized)
    if not ok:
        raise IOError(f"保存失败：{ai_image_path}")

    return True


def compute_diff_maps(real_image_path: str, generated_image_path: str, threshold: float) -> torch.Tensor:
    """
    计算像素级平均 RGB 差分并与阈值比较，返回二值 soft_map（0/1 的 torch.Tensor，HxW）。
    """
    real_bgr = cv2.imread(real_image_path, cv2.IMREAD_COLOR)
    gen_bgr = cv2.imread(generated_image_path, cv2.IMREAD_COLOR)
    if real_bgr is None or gen_bgr is None:
        raise FileNotFoundError(f"读取图像失败: {real_image_path} 或 {generated_image_path}")

    if real_bgr.shape[:2] != gen_bgr.shape[:2]:
        raise AssertionError("Input images must have the same dimensions (请先调用 sync_ai_image_size).")

    real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gen_rgb = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean_diff = np.abs(real_rgb - gen_rgb).mean(axis=2)  # HxW

    soft_bin = (mean_diff > float(threshold)).astype(np.uint8)  # 0/1
    soft_map = torch.from_numpy(soft_bin).to(torch.float32)
    return soft_map


def save_soft_map(soft_map: torch.Tensor, save_path: str):
    img = (soft_map * 255).byte().cpu().numpy()
    Image.fromarray(img).save(save_path)


def hash_string(s: str, length: int = 16) -> str:
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()[:length]


def hash_file(path: str, length: int = 16) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:length]


def load_descriptions(csv_path: str) -> dict:
    """
    加载 descriptions.csv 文件，返回一个字典：
    {
        "workspace": {
            "image_id": {
                "ann_id": "description text"
            }
        }
    }
    对于没有 ann_id 的情况，使用 "-1" 作为 key
    """
    descriptions = {}

    if not os.path.exists(csv_path):
        print(f"Warning: descriptions.csv not found at {csv_path}, text field will be empty.")
        return descriptions

    print(f"Loading descriptions from {csv_path}...")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            workspace = row['workspace']
            image_id = row['image_id']
            ann_id = row['ann_id']  # 可能是 "-1" 或具体值
            description = row['descriptions']

            if workspace not in descriptions:
                descriptions[workspace] = {}
            if image_id not in descriptions[workspace]:
                descriptions[workspace][image_id] = {}

            descriptions[workspace][image_id][ann_id] = description
            
    return descriptions


def get_description(descriptions: dict, workspace: str, image_id: str, ann_id: str = None) -> str:
    """
    从 descriptions 字典中获取对应的描述文本
    """
    if "qwen" in workspace:
        workspace = workspace[5:]
        
    if workspace not in descriptions:
        return ""

    if image_id not in descriptions[workspace]:
        return ""

    # 如果没有 ann_id，使用 "-1"
    lookup_ann_id = ann_id if ann_id else "-1"
    text = descriptions[workspace][image_id].get(lookup_ann_id, "")
    return text


def main():
    ap = argparse.ArgumentParser(description="Unified dataset construction script with text descriptions.")
    ap.add_argument("--id", required=True, help="Dataset ID under dataset-dir.")
    ap.add_argument("--dataset-dir", default="/workspace/dataset/raw_outputs_training", help="Path to the dataset directory.")
    ap.add_argument("--output-dir", default="/workspace/dataset/demo", help="Directory to save filtered dataset.")
    ap.add_argument("--tao", type=float, default=0.1, help="Threshold value for ground truth.")
    ap.add_argument("--dest-type", default="train", type=str, choices=["train", "test", "validation"], help="Destination split type.")
    ap.add_argument("--num-workers", type=int, default=None, help="Number of worker threads (default: 2 * CPU cores, capped at 32).")
    ap.add_argument("--anno", action="store_true", help="Enable annotation mode (nested name/ann_id structure).")
    ap.add_argument("--bg", action="store_true", help="Mark as background dataset (for tracking purposes).")
    ap.add_argument("--descriptions-csv", default="/home/jiacheng/Omni_detection/PIXAR/utils_preprocess/descriptions.csv", help="Path to descriptions.csv file.")
    ap.add_argument("--missing-csv-dir", default="./missing_text_logs", help="Directory to save the missing-text CSV report.")
    ap.add_argument("--missing-csv-name", default="missing_text.csv", help="Filename of the missing-text CSV report.")
    args = ap.parse_args()

    # 加载 descriptions
    descriptions = load_descriptions(args.descriptions_csv)

    # 修改 output_dir 以包含 tao 值
    args.output_dir = f"{args.output_dir}_{args.tao}"
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = os.path.join(args.dataset_dir, args.id)
    json_file = os.path.join(args.output_dir, "mapping.json")

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    if not os.path.exists(json_file):
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
        print(f"Created new mapping file: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        reverse_mapping = json.load(f)

    # 创建目录结构
    for split in ("train", "test", "validation"):
        os.makedirs(os.path.join(args.output_dir, split, "masks"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "tampered"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "soft_masks"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "real"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "full_synthetic"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "metadata"), exist_ok=True)

    # real 统一放在 train/real
    real_root = os.path.join(args.output_dir, "train", "real")
    os.makedirs(real_root, exist_ok=True)

    # 线程锁
    mapping_lock = threading.Lock()

    # worker 数
    if args.num_workers is not None and args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = os.cpu_count() or 4
        num_workers = min(32, num_workers * 2)

    print(f"Using {num_workers} worker threads.")
    print(f"Annotation mode: {'Enabled' if args.anno else 'Disabled'}")
    print(f"Background mode: {'Enabled' if args.bg else 'Disabled'}")

    dest_type = args.dest_type

    # 收集没有找到 text 的 entry
    missing_text_entries = []
    missing_text_lock = threading.Lock()

    # 根据 --anno 参数决定处理逻辑
    if args.anno:
        # ===== WITH ANNOTATION MODE =====
        # 数据结构: dataset_path/<name>/<ann_id>/
        all_names = sorted([n for n in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, n))])
        print(f"Found {len(all_names)} top-level names in dataset: {dataset_path}")

        # 只保存一次 real 的标记
        saved_real_flag = {}
        saved_real_lock = threading.Lock()

        def mark_saved_real(name: str) -> bool:
            """同一个 name 只允许保存一次 real。返回 True 表示这次需要保存。"""
            with saved_real_lock:
                if name in saved_real_flag:
                    return False
                saved_real_flag[name] = True
                return True

        # 构建 tasks
        tasks = []
        for name in all_names:
            name_dir = os.path.join(dataset_path, name)
            ann_ids = sorted([a for a in os.listdir(name_dir) if os.path.isdir(os.path.join(name_dir, a))])
            for ann_id in ann_ids:
                file_path = os.path.join(name_dir, ann_id)

                orig_mask_path = os.path.join(file_path, "mask.png")
                real_image_path = os.path.join(file_path, "original.png")
                ai_image_path = os.path.join(file_path, "generated.png")
                cls_info_path = os.path.join(file_path, "replace_info.json")

                if not (os.path.exists(real_image_path) and os.path.exists(ai_image_path) and os.path.exists(cls_info_path)):
                    continue

                tasks.append({
                    "name": name,
                    "ann_id": ann_id,
                    "file_path": file_path,
                    "orig_mask_path": orig_mask_path,
                    "real_image_path": real_image_path,
                    "ai_image_path": ai_image_path,
                    "cls_info_path": cls_info_path,
                    "entry": f"{name}/{ann_id}",
                })

        print(f"Found {len(tasks)} nested samples (name/ann_id).")

        def process_one_anno(task) -> int:
            name = task["name"]
            ann_id = task["ann_id"]
            entry = task["entry"]

            orig_mask_path = task["orig_mask_path"]
            real_image_path = task["real_image_path"]
            ai_image_path = task["ai_image_path"]
            cls_info_path = task["cls_info_path"]

            # 对齐尺寸
            try:
                _ = sync_ai_image_size(ai_image_path, real_image_path)
            except FileNotFoundError:
                tqdm.write(f"Skipping {entry} due to missing image.")
                return 0
            except Exception as e:
                tqdm.write(f"Skipping {entry} due to sync error: {e}")
                return 0

            # 计算 soft map
            try:
                soft_map = compute_diff_maps(real_image_path, ai_image_path, threshold=args.tao)
            except Exception as e:
                tqdm.write(f"Error computing diff for {entry}: {e}")
                return 0

            # 生成 ID
            tampered_id = hash_string(f"{args.id}/{name}/{ann_id}", length=16)
            try:
                real_id = hash_file(real_image_path, length=16)
            except Exception as e:
                tqdm.write(f"Error hashing real image for {entry}: {e}")
                return 0

            tampered_filename = f"tampered_{tampered_id}.png"
            tampered_mask_filename = f"tampered_{tampered_id}_mask.png"
            tampered_meta_filename = f"tampered_{tampered_id}_cls.json"
            real_filename = f"original_{real_id}.png"

            # 目标路径
            dst_ai_image = os.path.join(args.output_dir, dest_type, "tampered", tampered_filename)
            dst_soft_map = os.path.join(args.output_dir, dest_type, "soft_masks", tampered_mask_filename)
            dst_orig_mask = os.path.join(args.output_dir, dest_type, "masks", tampered_mask_filename)
            dst_meta = os.path.join(args.output_dir, dest_type, "metadata", tampered_meta_filename)
            dst_real_image = os.path.join(args.output_dir, dest_type, "real", real_filename)

            # 写 metadata (包含 cls 和 text)
            try:
                with open(cls_info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cls_info = data.get("replacement_categories", [])

                # 获取描述文本
                # ann_id 格式可能是 "ann_1227626_1438517"，需要去掉 "ann_" 前缀
                ann_id_for_lookup = ann_id.replace("ann_", "") if ann_id.startswith("ann_") else ann_id
                text_description = get_description(descriptions, args.id, name, ann_id_for_lookup)

                if not text_description:
                    with missing_text_lock:
                        missing_text_entries.append({
                            "dataset_id": args.id,
                            "image_id": name,
                            "ann_id": ann_id_for_lookup,
                            "entry": entry,
                        })

                with open(dst_meta, "w", encoding="utf-8") as wf:
                    json.dump({
                        "cls": cls_info,
                        "text": text_description
                    }, wf, ensure_ascii=False, indent=2)
            except Exception as e:
                tqdm.write(f"Error writing metadata for {entry}: {e}")
                return 0

            # 保存 tampered 图
            try:
                shutil.copy(ai_image_path, dst_ai_image)
            except Exception as e:
                tqdm.write(f"Error copying tampered image for {entry}: {e}")
                return 0

            # 保存 soft map
            try:
                save_soft_map(soft_map, dst_soft_map)
            except Exception as e:
                tqdm.write(f"Error saving soft map for {entry}: {e}")
                return 0

            # 保存原 mask
            try:
                if not os.path.exists(orig_mask_path):
                    save_soft_map(soft_map, dst_orig_mask)
                else:
                    shutil.copy(orig_mask_path, dst_orig_mask)
            except Exception as e:
                tqdm.write(f"Error saving mask for {entry}: {e}")
                return 0

            # 更新 mapping
            with mapping_lock:
                reverse_mapping[tampered_filename] = {
                    "entry": entry,
                    "real": real_filename,
                    "type": args.id,
                    "bg": args.bg
                }

            # 只对同一个 name 保存一次 real
            try:
                if mark_saved_real(name):
                    if not os.path.exists(dst_real_image):
                        shutil.copy(real_image_path, dst_real_image)
            except Exception as e:
                tqdm.write(f"Error copying real image for {entry}: {e}")

            return 1

        # 多线程处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(
                    executor.map(process_one_anno, tasks),
                    total=len(tasks),
                    desc="Processing samples",
                    unit="item",
                )
            )

    else:
        # ===== WITHOUT ANNOTATION MODE =====
        # 数据结构: dataset_path/<name>/
        all_names = sorted([n for n in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, n))])
        print(f"Found {len(all_names)} samples in dataset: {dataset_path}")

        def process_one_no_anno(name: str) -> int:
            file_path = os.path.join(dataset_path, name)
            entry = f"{name}"

            orig_mask_path = os.path.join(file_path, "mask.png")
            real_image_path = os.path.join(file_path, "original.png")
            ai_image_path = os.path.join(file_path, "generated.png")
            cls_info_path = os.path.join(file_path, "replace_info.json")

            # 对齐尺寸
            try:
                _ = sync_ai_image_size(ai_image_path, real_image_path)
            except FileNotFoundError:
                tqdm.write(f"Skipping {entry} due to missing image.")
                return 0

            if not (os.path.exists(ai_image_path) and os.path.exists(real_image_path) and os.path.exists(cls_info_path)):
                tqdm.write(f"Skipping {entry} due to missing original image or metadata.")
                return 0

            # 计算 soft map
            try:
                soft_map = compute_diff_maps(real_image_path, ai_image_path, threshold=args.tao)
            except Exception as e:
                tqdm.write(f"Error computing diff for {entry}: {e}")
                return 0

            # 生成 ID
            tampered_id = hash_string(f"{args.id}/{name}", length=16)
            try:
                real_id = hash_file(real_image_path, length=16)
            except Exception as e:
                tqdm.write(f"Error hashing real image for {entry}: {e}")
                return 0

            tampered_filename = f"tampered_{tampered_id}.png"
            tampered_mask_filename = f"tampered_{tampered_id}_mask.png"
            tampered_meta_filename = f"tampered_{tampered_id}_cls.json"
            real_filename = f"original_{real_id}.png"

            # 目标路径
            dst_ai_image = os.path.join(args.output_dir, dest_type, "tampered", tampered_filename)
            dst_soft_map = os.path.join(args.output_dir, dest_type, "soft_masks", tampered_mask_filename)
            dst_orig_mask = os.path.join(args.output_dir, dest_type, "masks", tampered_mask_filename)
            dst_meta = os.path.join(args.output_dir, dest_type, "metadata", tampered_meta_filename)
            dst_real_image = os.path.join(args.output_dir, dest_type, "real", real_filename)

            # 写 metadata (包含 cls 和 text)
            try:
                with open(cls_info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cls_info = data.get("replacement_categories", [])

                # 获取描述文本（wo-anno 使用 "-1" 作为 ann_id）
                text_description = get_description(descriptions, args.id, name, "-1")

                if not text_description:
                    with missing_text_lock:
                        missing_text_entries.append({
                            "dataset_id": args.id,
                            "image_id": name,
                            "ann_id": "-1",
                            "entry": entry,
                        })

                with open(dst_meta, "w", encoding="utf-8") as wf:
                    json.dump({
                        "cls": cls_info,
                        "text": text_description
                    }, wf, ensure_ascii=False, indent=2)
            except Exception as e:
                tqdm.write(f"Error writing metadata for {entry}: {e}")
                return 0

            # 保存 tampered 图
            try:
                shutil.copy(ai_image_path, dst_ai_image)
            except Exception as e:
                tqdm.write(f"Error copying tampered image for {entry}: {e}")
                return 0

            # 保存 soft map
            try:
                save_soft_map(soft_map, dst_soft_map)
            except Exception as e:
                tqdm.write(f"Error saving soft map for {entry}: {e}")
                return 0

            # 保存原 mask
            try:
                if not os.path.exists(orig_mask_path):
                    save_soft_map(soft_map, dst_orig_mask)
                else:
                    shutil.copy(orig_mask_path, dst_orig_mask)
            except Exception as e:
                tqdm.write(f"Error saving mask for {entry}: {e}")
                return 0

            # 更新 mapping
            with mapping_lock:
                reverse_mapping[tampered_filename] = {
                    "entry": entry,
                    "real": real_filename,
                    "type": args.id,
                    "bg": args.bg
                }

            # 保存 real 图
            try:
                if not os.path.exists(dst_real_image):
                    shutil.copy(real_image_path, dst_real_image)
            except Exception as e:
                tqdm.write(f"Error copying real image for {entry}: {e}")

            return 1

        # 多线程处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(
                    executor.map(process_one_no_anno, all_names),
                    total=len(all_names),
                    desc="Processing samples",
                    unit="img",
                )
            )

    # 写回 mapping.json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(reverse_mapping, f, ensure_ascii=False, indent=2)

    # 写出 missing-text CSV
    os.makedirs(args.missing_csv_dir, exist_ok=True)
    missing_csv_path = os.path.join(args.missing_csv_dir, args.missing_csv_name)
    file_exists = os.path.exists(missing_csv_path)
    with open(missing_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset_id", "image_id", "ann_id", "entry"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(missing_text_entries)
    print(f"Missing-text report saved to: {missing_csv_path} ({len(missing_text_entries)} entries)")

    print("Done.")


if __name__ == "__main__":
    main()
