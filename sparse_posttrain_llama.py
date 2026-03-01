"""
Author: Jonathan Svirsky, 2026
Based on the training code from Galore: https://github.com/jiaweizzhao/GaLore
"""
import os
import time
import json
import random
import argparse
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer,  LlamaForCausalLM as HF_LlamaForCausalLM
import datasets
import datasets.distributed
import wandb
from tqdm import tqdm
from loguru import logger
from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
import bitsandbytes as bnb
from model import SparseLayer
import safetensors.torch 
transformers.logging.set_verbosity_error()


def set_trainable(model, trainable_substrings, train_bias="none"):
    for n, p in model.named_parameters():
        p.requires_grad = any(s in n for s in trainable_substrings)

    if train_bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif train_bias == "sparse_only":
        for m in model.modules():
            if isinstance(m, SparseLayer) and getattr(m, "bias", None) is not None:
                m.bias.requires_grad = True


def put_frozen_weights_to_cpu(model):
    for m in model.modules():
        if isinstance(m, SparseLayer):
            m.weight_ = m.weight_.cpu()  # Assuming SparseLayer has a to_cpu() method


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")    
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--ckpt_root_dir", type=str, default="checkpoints2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.5)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--num_training_steps", type=int, default=10000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default='2B',
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1948)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    parser.add_argument("--beta1", type=float, default=0.0)
    
    parser.add_argument("--target_sparsity", type=float, default=0.3)
    parser.add_argument("--kurt", action="store_true")
    parser.add_argument("--sparsity_start_step", type=int, default=0) # equal to warmup steps
    parser.add_argument("--sparsity_lambda", type=float, default=50)
    parser.add_argument("--disable_sparsity", action="store_true")
    parser.add_argument("--bias", action="store_true")

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=True, type=bool)
    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


def count_parameters(model):
    total_params = 0
    non_classifier_params = 0
    bias_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        total_params += parameter.numel()
        if "score" not in name and "bias" not in name:
            non_classifier_params += parameter.numel()
        if "bias" in name:
            bias_params += parameter.numel()

    logger.info(f"Total Trainable Params: {total_params}")
    logger.info(f"Total Non Classifier Params: {non_classifier_params}")
    logger.info(f"Total Bias Params: {bias_params}")

    return total_params


def count_effective_total_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    # SparseLayer stores frozen base weights in `weight_` (not a registered Parameter).
    for module in model.modules():
        if isinstance(module, SparseLayer):
            total_params += module.weight_.numel()
    return total_params


def compute_compressed_parameters(sparse_layers):
    total = 0
    for m in sparse_layers:
        total += m.number_compressed_parameters()
    return total


def compute_sparsity_loss(sparse_layers):
    return torch.stack([m.sparsity_loss() for m in sparse_layers]).mean()
   

@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True) #DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches
    if not args.single_gpu:
        # Gather losses across all GPUs
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
        dist.all_gather(gathered_losses, total_loss)
        total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def get_most_recent_dir(parent_dir):
    # List only directories inside parent_dir
    dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
            if os.path.isdir(os.path.join(parent_dir, d))]
    
    if not dirs:
        return None  # No subdirectories

    # Sort by modification time (latest first)
    most_recent = max(dirs, key=os.path.getmtime)
    return most_recent

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # assume single GPU training for now, can add distributed training later
    global_rank = 0
    local_rank = 0
    world_size = 1
    torch.cuda.set_device(local_rank)
    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")
    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project=f"llama7b-c4-gates")
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    logger.info(f"Shuffling data with seed {args.seed }")
    data: datasets.Dataset = data.shuffle(seed=args.seed )
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    # tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
    model = HF_LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype='auto')
    model_config = model.config
   
    total_params = 0
    for name, parameter in model.named_parameters():
        total_params += parameter.numel()
    logger.info(f"Total parameters in the model: {total_params / 1_000_000:.2f}M")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
    if not args.disable_sparsity:        
        replace_linear_with_sparse(model)
    if args.bias:
        set_trainable(
            model, 
            trainable_substrings=["gates_rows", "gates_columns"], 
            train_bias="sparse_only"
        )
    else:
        set_trainable(
            model, 
            trainable_substrings=["gates_rows", "gates_columns"], 
            train_bias='none'
        )

    # the actual number of parameters is larger since the sparse 
    # layers include the original parameters together with new trainable gates and biases
    count_parameters(model)

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is None:
        # try to load the latest checkpoint from save_dir
        # get the most updated dir from directory:
        # List only directories inside parent_dir
        most_updated_model_dir = get_most_recent_dir(args.ckpt_root_dir) if os.path.exists(args.ckpt_root_dir) else None
        if most_updated_model_dir is None:
            logger.warning(f"No checkpoints found in {args.ckpt_root_dir}, starting from scratch")
            args.continue_from = None
        else:
            last_ckpt_dir = get_most_recent_dir(most_updated_model_dir)
            args.continue_from =  last_ckpt_dir if last_ckpt_dir is not None else None

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.continue_from, "model.safetensors")
            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Model successfully loaded (strict=True policy)")
        
        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    n_total_params = count_effective_total_parameters(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
   
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'allenai/c4',
        "model": args.model_name,
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    # print params and trainable params
    # logger.info(f"\n{model}\n")
    logger.info(f"Total params (effective): {n_total_params / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    # pad_idx = tokenizer.pad_token_id
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    pad_idx = 0
    model.generation_config.pad_token_id = pad_idx
    tokenizer.pad_token_id =  0
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    if args.continue_from is not None:
        # ---- Load optimizer & scheduler state (includes LR) ----
        optimizer_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(optimizer_path):
            logger.info(f"Loading optimizer and scheduler from {optimizer_path}")
            optimizer_checkpoint = torch.load(optimizer_path, map_location="cpu")

            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
            scheduler.load_state_dict(optimizer_checkpoint["scheduler"])

            # Restore steps from optimizer checkpoint (overrides training_state.json if needed)
            global_step = optimizer_checkpoint.get("global_step", 0)
            update_step = optimizer_checkpoint.get("update_step", 0)

            # Log restored LRs
            for i, pg in enumerate(optimizer.param_groups):
                logger.info(f"Restored LR for param group {i}: {pg['lr']}")
        else:
            logger.warning("No optimizer checkpoint found — starting optimizer/scheduler fresh")

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    sparse_layers = [m for m in model.modules() if isinstance(m, SparseLayer)]

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            logger.info(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
        model_loss = model(**batch, labels=labels).loss
        sparsity_lambda = 0 if update_step < args.sparsity_start_step else args.sparsity_lambda
        sparsity_loss = compute_sparsity_loss(sparse_layers)        
        loss = model_loss + sparsity_lambda * sparsity_loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        if global_step % args.gradient_accumulation != 0:
            continue
        # The below code is only executed during the update step
        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            model.save_pretrained(current_model_directory)
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
                
            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            
            compression = compute_compressed_parameters(sparse_layers)
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    "compressed_params": compression,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}, compressed_params: {compression / 1000000:.02f}M")
        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "model_loss": model_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                },
                step=global_step,
            )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(current_model_directory)
        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")
        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    logger.info(f"Rank {global_rank} finished successfully")


def replace_linear_with_sparse(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(module, name, SparseLayer(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    original_layer=child,
                    dtype=args.dtype,
                    target_sparsity=args.target_sparsity,
                    kurt=args.kurt,
                    bias=args.bias,
                    ))
                else:
                    replace_linear_with_sparse(child)


if __name__ == "__main__":
    logger.info("Starting script")
    args = parse_args(None)
    main(args)
