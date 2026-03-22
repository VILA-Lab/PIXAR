import argparse
import os
import shutil
import sys
import time
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.PIXAR import PIXARForCausalLM
from model.llava import conversation as conversation_lib
from utils.PIXAR_Set import collate_fn, CustomDataset
from utils.batch_sampler import BatchSampler
import torch.distributed as dist
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def parse_args(args):
    parser = argparse.ArgumentParser(description="PIXAR Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="pixar", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)

    # Add Stage-specific arguments
    parser.add_argument("--num_classes", type=int, default=3,
                       help="Number of classes for classification in stage 1")
    parser.add_argument("--use_stage1_cls", action="store_true", default=True,
                   help="Whether to use Stage 1 CLS token in Stage 2")
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)
    parser.add_argument("--text_loss_weight", default=1.0, type=float, help="Weight for text generation loss")
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--num_saves", default=10, type=int,
                        help="Number of evenly-spaced checkpoints (and validations) during training")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    
    # 在 parse_args(...) 里其它 OBJ 参数后面追加
    parser.add_argument("--obj_pos_weight", type=float, default=60.0,
                        help="Scalar pos_weight for BCEWithLogits (None => auto-compute from batch).")
    parser.add_argument("--obj_pos_weight_max", type=float, default=100.0,
                        help="Clamp upper bound for auto-computed pos_weight.")
    print('add pos weight args done')

    # new arguments for object classification head
    parser.add_argument("--num_obj_classes", type=int, default=81,
                        help="Number of object categories for <OBJ> image-level classification")
    parser.add_argument("--obj_loss_weight", type=float, default=1.0,
                        help="Loss weight for <OBJ> image-level classification head")
    
    parser.add_argument("--obj_threshold", type=float, default=0.5,
                        help="Threshold for multi-label prediction on OBJ head")
    parser.add_argument("--log_obj_prefix", type=str, default="obj",
                        help="TensorBoard tag prefix for OBJ multi-label metrics")
    
    parser.add_argument(
        "--seg_prompt_mode",
        type=str,
        default="fuse",
        choices=["seg_only", "fuse", "text_only"],
        help="SAM prompt embedding mode for segmentation ablation."
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="ours",
        choices=["ours", "others"],
        help="Mask type for loss computation: 'ours' uses gt_soft_mask, 'others' uses gt_mask."
    )


    return parser.parse_args(args)
def main(args):
    args = parse_args(args)
    # Move the check here, after parsing
    deepspeed.init_distributed()
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        print("========== Hyperparameters ==========")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")
        print("=====================================")
    else:
        writer = None

     # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_token = tokenizer.add_tokens("[CLS]")
    num_added_token = tokenizer.add_tokens("[SEG]")
    # === NEW: add <OBJ> token ===
    num_added_token = tokenizer.add_tokens("[OBJ]")
    # === NEW: add <END> token ===
    num_added_token = tokenizer.add_tokens("[END]")

    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    # === NEW: record <OBJ> token index ===
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]
    # === NEW: record <END> token index ===
    args.end_token_idx = tokenizer("[END]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "text_loss_weight": args.text_loss_weight,  # NEW: text loss weight
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "obj_token_idx": args.obj_token_idx,
        "num_obj_classes": args.num_obj_classes,
        "obj_loss_weight": args.obj_loss_weight,
        "obj_pos_weight": args.obj_pos_weight,
        "obj_pos_weight_max": args.obj_pos_weight_max,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_prompt_mode": args.seg_prompt_mode,
        "mask_type": args.mask_type,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = PIXARForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("\nChecking specific components:")
    for component in [ "cls_head", "pixar_fc1", "attention_layer", "text_hidden_fcs"]:
        matching_params = [n for n, _ in model.named_parameters() if component in n]
        if matching_params:
            print(f"Found {component} in parameters: {matching_params}")
        else:
            print(f"Component not found: {component}")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_pixar_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False


    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
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
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
                model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["embed_tokens", "mask_decoder", "cls_head", "obj_head", "seg_proj", "text_proj", "gate_mlp"]
            ]
        ):
            p.requires_grad = True

    print("Checking trainable parameters:")
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"Trainable: {n} with {p.numel()} parameters")
            total_params += p.numel()
    print(f"Total trainable parameters: {total_params}")

    world_size = dist.get_world_size()
    args.distributed = world_size > 1
    train_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,  # Root directory containing image data
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,  # Vision model used for pre-processing (e.g., CLIP)
        split="train",  # Specify that this is the training split
        precision=args.precision,  # Precision for image processing
        image_size=args.image_size,  # Image size for resizing

    )
    print(f"\nInitializing datasets:")
    print(f"Training split size: {len(train_dataset)}")

    if args.no_eval == False:
        val_dataset = CustomDataset(
            base_image_dir=args.dataset_dir,  # Root directory containing image data
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,  # Vision model used for pre-processing (e.g., CLIP)
            split="validation",  # Specify that this is the training split
            precision=args.precision,  # Precision for image processing
            image_size=args.image_size,  # Image size for resizing
    )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0,  # Dynamic loss scaling
            "initial_scale_power": 12,  # Start lower (2^12 = 4096)
            "loss_scale_window": 1000,
            "min_loss_scale": 1,
            "hysteresis": 2
        },
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    batch_sampler = BatchSampler(
        dataset=train_dataset,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        world_size=dist.get_world_size(),
        rank=args.local_rank
    )

    # Create DataLoader with BatchSampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            cls_token_idx=args.cls_token_idx,
            obj_token_idx=args.obj_token_idx,
            seg_token_idx=args.seg_token_idx,
        ),
    )
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=None,  # Set to None since we're providing our own loader
    )

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir,  "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        val_sampler = BatchSampler(
            dataset=val_dataset,
            batch_size=args.val_batch_size,
            world_size=dist.get_world_size(),
            rank=args.local_rank
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=partial(
                 collate_fn,
                 tokenizer=tokenizer,
                 conv_type=args.conv_type,
                 use_mm_start_end=args.use_mm_start_end,
                 local_rank=args.local_rank,
                 cls_token_idx=args.cls_token_idx,
                 obj_token_idx=args.obj_token_idx,
                 seg_token_idx=args.seg_token_idx,
             ),
        )

    train_iter = iter(train_loader)

    best_acc, best_score, cur_ciou = 0.0, 0.0, 0.0

    if args.eval_only:
        acc, giou, ciou, _ = validate(val_loader, model_engine, 0, writer, args)  # Classification validation
        exit()

    num_saves = args.num_saves
    step = max(1, args.epochs // num_saves)
    validation_epochs = list(range(step, args.epochs, step))
    if args.epochs not in validation_epochs:
        validation_epochs.append(args.epochs)  # always include last epoch
    if args.local_rank == 0:
        print(f"\nTraining Configuration:")
        print(f"Total epochs: {args.epochs}")
        print(f"Validation will be performed after epochs: {validation_epochs}")
    for epoch in range(args.start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        if (epoch + 1) in validation_epochs:  # +1 because epoch starts from 0
            if args.local_rank == 0:
                print(f"\nPerforming validation after epoch {epoch + 1}")

            if args.no_eval == False:
                acc, giou, ciou, _ = validate(val_loader, model_engine, epoch, writer, args)
                is_best_iou = giou > best_score
                is_best_acc = acc > best_acc
                best_score = max(giou, best_score)
                best_acc = max(acc, best_acc)
                cur_ciou = ciou if is_best_iou else cur_ciou
                cur_acc = acc if is_best_acc else cur_acc
                is_best = is_best_iou or is_best_acc
            else:
                acc, giou, ciou = -1.0, -1.0, -1.0
                is_best = False

            if args.local_rank == 0:
                print(f"Current accuracy: {acc:.2f}%, Best accuracy: {best_acc:.2f}%")
                print(f"Current iou: {cur_ciou:.2f}%, Best score: {best_score:.2f}%")
            # Save checkpoints for best performance
            if args.no_eval or is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save(
                                {"epoch": epoch},
                                os.path.join(
                                    args.log_dir,
                                    f"meta_log_acc{best_acc:.3f}_iou{best_score:.3f}.pth"
                                ),
                    )
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir)
                torch.distributed.barrier()
        else:
            if args.local_rank == 0:
                print(f"Epoch {epoch + 1} completed. Skipping validation.")

        # Save final epoch regardless of validation
        if epoch == args.epochs - 1:
            save_dir = os.path.join(args.log_dir, "final_checkpoint")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                print(f"\nTraining completed. Final checkpoint saved to {save_dir}")

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("ClsLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    obj_losses = AverageMeter("ObjLoss", ":.4f")
    text_losses = AverageMeter("TextLoss", ":.4f")  # NEW: text loss meter
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, cls_losses, mask_bce_losses, mask_dice_losses, mask_losses, obj_losses, text_losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    model.train()
    end = time.time()
    # total micro-batches per epoch = steps_per_epoch * grad_accumulation_steps
    # DeepSpeed handles gradient accumulation internally: it accumulates gradients
    # over grad_accumulation_steps micro-batches and then does all-reduce + optimizer step.
    total_micro_steps = args.steps_per_epoch * args.grad_accumulation_steps
    for global_step in range(total_micro_steps):
        try:
            input_dict = next(train_iter)
        except:
            train_iter = iter(train_loader)
            input_dict = next(train_iter)

        data_time.update(time.time() - end)
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        output_dict = model(**input_dict)
        loss = output_dict["loss"]
        cls_loss = output_dict["cls_loss"]
        mask_bce_loss = output_dict["mask_bce_loss"]
        mask_dice_loss = output_dict["mask_dice_loss"]
        mask_loss = output_dict["mask_loss"]
        obj_loss = output_dict.get("obj_loss", torch.tensor(0.0, device=loss.device))
        text_loss = output_dict.get("text_loss", torch.tensor(0.0, device=loss.device))
        losses.update(loss.item(), input_dict["images"].size(0))
        cls_losses.update(cls_loss.item(), input_dict["images"].size(0))
        if input_dict['cls_labels'][0] == 2:
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
        if "obj_labels" in input_dict and input_dict["obj_labels"].numel() > 0:
            n_obj = input_dict["obj_labels"].shape[0]
            obj_losses.update(obj_loss.item(), max(n_obj, 1))
        if text_loss.item() > 0:
            text_losses.update(text_loss.item(), input_dict["images"].size(0))

        model.backward(loss)
        model.step()

        # Log at optimizer step boundaries (every grad_accumulation_steps micro-batches)
        optimizer_step = global_step // args.grad_accumulation_steps
        is_optimizer_step = (global_step + 1) % args.grad_accumulation_steps == 0

        if is_optimizer_step:
            batch_time.update(time.time() - end)
            end = time.time()

            if optimizer_step % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()
                    losses.all_reduce()
                    cls_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()
                    obj_losses.all_reduce()
                    text_losses.all_reduce()

                if args.local_rank == 0:
                    progress.display(optimizer_step + 1)
                    writer.add_scalar("train/loss", losses.avg, optimizer_step)
                    writer.add_scalar("train/cls_loss", cls_losses.avg, optimizer_step)
                    writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, optimizer_step)
                    writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, optimizer_step)
                    writer.add_scalar("train/mask_loss", mask_losses.avg, optimizer_step)
                    writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, optimizer_step)
                    writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, optimizer_step)
                    writer.add_scalar("train/obj_loss", obj_losses.avg, optimizer_step)
                    writer.add_scalar("train/text_loss", text_losses.avg, optimizer_step)
                batch_time.reset()
                data_time.reset()
                losses.reset()
                cls_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()
                obj_losses.reset()
                text_losses.reset()

            if optimizer_step != 0:
                curr_lr = scheduler.get_last_lr()
                if args.local_rank == 0:
                    writer.add_scalar("train/lr", curr_lr[0], optimizer_step)

    return train_iter
import random

def validate(val_loader, model_engine, epoch, writer, args, sample_ratio=None):
    """
    Validate the model with option for random sampling
    Args:
        sample_ratio: if None, use all data; if float (e.g., 0.1), randomly sample that portion
    """
    model_engine.eval()
    correct = 0
    total = 0
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device='cuda')
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    # --- NEW: Multi-label accumulators for OBJ ---
    obj_tp_total = 0.0      # micro-averaged
    obj_fp_total = 0.0
    obj_fn_total = 0.0
    obj_exact_match_total = 0  # subset accuracy（完全一致）
    obj_rows_total = 0         # N_obj 累计
    # per-class 累计（macro）
    obj_tp_per_class = None  # 按列累计，后面根据 K 初始化
    obj_fp_per_class = None
    obj_fn_per_class = None

    # Calculate total number of batches and samples to use
    total_batches = len(val_loader)
    if sample_ratio is not None:
        num_batches = max(1, int(total_batches * sample_ratio))
        # Generate random indices for sampling
        sample_indices = set(random.sample(range(total_batches), num_batches))
        print(f"\nValidating on {num_batches}/{total_batches} randomly sampled batches...")

    for batch_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        # Skip batches not in our sample if sampling is enabled
        if sample_ratio is not None and batch_idx not in sample_indices:
            continue
        if batch_idx == 0:
            print("\nFirst validation batch details:")
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"{key} length: {len(value)}")

        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        # Debug first processed batch
        if total == 0:
            print("\nProcessing first batch:")
            print("Input dict keys:", input_dict.keys())

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        input_dict['inference'] = True
        obj_correct, obj_total = 0, 0
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # Get predictions
        logits = output_dict["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        cls_labels = input_dict["cls_labels"]
        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)
        
        # === <OBJ> Multi-label metrics (only for tampered samples) ===
        if ("obj_logits" in output_dict) and ("obj_labels" in input_dict):
            tampered_mask = (cls_labels == 2)
            if tampered_mask.any():
                gt = input_dict["obj_labels"][tampered_mask]  # [N_tampered, K]
                logits_obj = output_dict["obj_logits"][tampered_mask]  # [N_tampered, K]
                probs = logits_obj.sigmoid()
                pred = (probs >= args.obj_threshold).to(gt.dtype)

                if obj_tp_per_class is None:
                    K = gt.shape[1]
                    device = gt.device
                    obj_tp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
                    obj_fp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
                    obj_fn_per_class = torch.zeros(K, device=device, dtype=torch.float64)
                tp = (pred * gt).sum().double()
                fp = (pred * (1 - gt)).sum().double()
                fn = ((1 - pred) * gt).sum().double()
                obj_tp_total += tp.item()
                obj_fp_total += fp.item()
                obj_fn_total += fn.item()

                exact_match = (pred == gt).all(dim=1).sum().item()
                obj_exact_match_total += exact_match
                obj_rows_total += gt.shape[0]

                obj_tp_per_class += (pred * gt).sum(dim=0).double()
                obj_fp_per_class += (pred * (1 - gt)).sum(dim=0).double()
                obj_fn_per_class += ((1 - pred) * gt).sum(dim=0).double()

        for t, p in zip(cls_labels, preds):
            confusion_matrix[t.long(), p.long()] += 1
        # Debug first batch predictions
        # Segmentation validation (only for "object/part synthetic" images, cls_label == 2)
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_soft_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    # Reduce and calculate final metrics
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if len(acc_iou_meter.avg) > 1 else 0.0

    # Calculate classification accuracy
    accuracy = correct / total * 100.0
    confusion_matrix = confusion_matrix.cpu()
    class_names = ['Real', 'Full Synthetic', 'Tampered']
    per_class_metrics = {}
    for i in range(num_classes):
        tp = confusion_matrix[i, i]  # Diagonal elements are true positives
        fp = confusion_matrix[:, i].sum() - tp  # Column sum minus TP = false positives
        fn = confusion_matrix[i, :].sum() - tp  # Row sum minus TP = false negatives
        tn = confusion_matrix.sum() - (tp + fp + fn)  # Rest are true negatives

        # Total samples of this class (row sum)
        total_class_samples = confusion_matrix[i, :].sum()

        # Metrics calculations
        class_accuracy = float(tp / total_class_samples) if total_class_samples > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[class_names[i]] = {
            'accuracy': class_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Calculate pixel accuracy
    pixel_correct = intersection_meter.sum[1]  # Correctly classified pixels (excluding background)
    pixel_total = union_meter.sum[1]  # Total pixels (excluding background)
    pixel_accuracy = pixel_correct / (pixel_total + 1e-10) * 100.0

    iou = ciou  # Use ciou as the IoU for the foreground class
    f1_score = 2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10) if (iou + accuracy / 100) > 0 else 0.0

    # Calculate average precision and recall for AUC approximation
    avg_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])

    # Approximate AUC as the area under the average precision-recall curve
    auc_approx = avg_precision * avg_recall
    
    # --- NEW: finalize OBJ multi-label metrics ---
    # micro
    obj_micro_prec = obj_tp_total / (obj_tp_total + obj_fp_total + 1e-12) if (obj_tp_total + obj_fp_total) > 0 else 0.0
    obj_micro_rec  = obj_tp_total / (obj_tp_total + obj_fn_total + 1e-12) if (obj_tp_total + obj_fn_total) > 0 else 0.0
    obj_micro_f1   = (2 * obj_micro_prec * obj_micro_rec / (obj_micro_prec + obj_micro_rec + 1e-12)) if (obj_micro_prec + obj_micro_rec) > 0 else 0.0
    # subset accuracy
    obj_subset_acc = (obj_exact_match_total / obj_rows_total) if obj_rows_total > 0 else 0.0
    # macro（逐类平均）
    if obj_tp_per_class is not None:
        prec_c = obj_tp_per_class / (obj_tp_per_class + obj_fp_per_class + 1e-12)
        rec_c  = obj_tp_per_class / (obj_tp_per_class + obj_fn_per_class + 1e-12)
        f1_c   = (2 * prec_c * rec_c / (prec_c + rec_c + 1e-12))
        obj_macro_prec = float(prec_c.mean().item())
        obj_macro_rec  = float(rec_c.mean().item())
        obj_macro_f1   = float(f1_c.mean().item())
    else:
        obj_macro_prec = obj_macro_rec = obj_macro_f1 = 0.0
    
    # Log metrics
    if args.local_rank == 0:
        writer.add_scalar("val/accuracy", accuracy, epoch)
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/pixel_accuracy", pixel_accuracy, epoch)
        writer.add_scalar("val/iou", iou, epoch)
        writer.add_scalar("val/f1_score", f1_score, epoch)
        writer.add_scalar("val/auc_approx", auc_approx, epoch)
        # --- NEW: write multi-label OBJ metrics ---
        pfx = args.log_obj_prefix
        writer.add_scalar(f"val/{pfx}_micro_precision", obj_micro_prec, epoch)
        writer.add_scalar(f"val/{pfx}_micro_recall",    obj_micro_rec, epoch)
        writer.add_scalar(f"val/{pfx}_micro_f1",        obj_micro_f1, epoch)
        writer.add_scalar(f"val/{pfx}_subset_acc",      obj_subset_acc, epoch)
        writer.add_scalar(f"val/{pfx}_macro_precision", obj_macro_prec, epoch)
        writer.add_scalar(f"val/{pfx}_macro_recall",    obj_macro_rec, epoch)
        writer.add_scalar(f"val/{pfx}_macro_f1",        obj_macro_f1, epoch)
        for class_name, metrics in per_class_metrics.items():
         for metric_name, value in metrics.items():
             writer.add_scalar(f"val/{class_name.lower().replace('/', '_')}_{metric_name}", value, epoch)

        validation_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100}%)"
        print(f"\n{validation_type} Validation Results:")
        print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}%")
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
        print(f"IoU: {iou:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Approximate AUC: {auc_approx:.4f}")
        print(f"Total correct classifications: {correct}")
        print(f"Total classification samples: {total}")
        print("\n[OBJ] Multi-Label Metrics:")
        print(f"  threshold: {args.obj_threshold:.2f}")
        print(f"  micro  - P: {obj_micro_prec:.4f}, R: {obj_micro_rec:.4f}, F1: {obj_micro_f1:.4f}")
        print(f"  macro  - P: {obj_macro_prec:.4f}, R: {obj_macro_rec:.4f}, F1: {obj_macro_f1:.4f}")
        print(f"  subset - Acc: {obj_subset_acc:.4f}")
        print("\nPer-Class Metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"\n{class_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        print("\nConfusion Matrix:")
        print("Predicted ")
        print("Actual ")
        print(f"{'':20}", end="")  # Add initial spacing
        for name in class_names:
            print(f"{name:>12}", end="")  # Align class names
        print()  # New line

        for i, class_name in enumerate(class_names):
            print(f"{class_name:20}", end="")  # Left align class names with fixed width
            for j in range(num_classes):
                print(f"{confusion_matrix[i, j]:12.0f}", end="")
            print()  # New line

    return accuracy, giou, ciou, per_class_metrics

if __name__ == "__main__":
    main(sys.argv[1:])
