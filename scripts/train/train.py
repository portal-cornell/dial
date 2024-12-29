from typing import Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
import os

from dial.arguments import Arguments
from dial.model import Model
from transformers import HfArgumentParser
from accelerate import Accelerator
from dial.common import initialize
from dial.utils import setup_wandb, seed
from scripts.evaluate import evaluate

torch.set_float32_matmul_precision("high")

def train(model: Model, 
          optimizer: torch.optim.Optimizer, 
          dataloader: torch.utils.data.DataLoader, 
          loss_fn: Callable, 
          args: Arguments, 
          accelerator: Accelerator,
          eval_dataloader: torch.utils.data.DataLoader = None,
          eval_loss_fn: Callable = None):
    checkpoint_dir = os.path.join(args.root_dir, "checkpoints")
    global_step = 0
    if eval_dataloader is not None:
        eval_metrics = evaluate(model, eval_dataloader, eval_loss_fn, args, accelerator, global_step)
        print(f"Start eval metrics:")
        for key, value in eval_metrics.items():
            print(f"{key}: {value}")

    model.save_pretrained(os.path.join(checkpoint_dir, f"epoch_pre"))
    model.global_step = global_step
    print(f"clip grad norm, {args.clip_grad_norm}")
    for epoch in range(args.num_epochs):
        seed(args)
        model.mode = "train"
        if args.train_mode in ["domain_adaptation", "domain_adaptation_contrastive", "domain_adaptation_mcq", "oracle"] and args.shuffle_target_dataset and args.dataset_type != "domain_adaptation_wrapped":
            dataloader.dataset.shuffle()
        model.train()
        pbar = tqdm.tqdm(dataloader)
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            with accelerator.autocast():
                loss, extra_metrics, outputs = loss_fn(batch, model, args)
                accelerator.backward(loss)

                if args.clip_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                if args.gradient_accumulation_steps is None or (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if args.use_wandb:
                extra_metrics = {f"train/{k}": v for k, v in extra_metrics.items()}
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    **extra_metrics
                })
            
            global_step += 1
            model.global_step = global_step

            pbar.set_description(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.3f}")

            if (step + 1) % args.step_eval_freq == 0 and eval_dataloader is not None:
                eval_metrics = evaluate(model, eval_dataloader, eval_loss_fn, args, accelerator, global_step)
                print(f"Epoch {epoch} step {step} eval metrics:")
                for key, value in eval_metrics.items():
                    print(f"{key}: {value}")

            if (step + 1) % args.step_save_freq == 0:
                model.save_pretrained(os.path.join(checkpoint_dir, f"epoch_{epoch}_step_{step+1}"))
        
        if (epoch + 1) % args.epoch_eval_freq == 0 and eval_dataloader is not None:
            eval_metrics = evaluate(model, eval_dataloader, eval_loss_fn, args, accelerator, global_step)
            print(f"Epoch {epoch} eval metrics:")
            for key, value in eval_metrics.items():
                print(f"{key}: {value}")

        if (epoch + 1) % args.epoch_save_freq == 0:
            model.save_pretrained(os.path.join(checkpoint_dir, f"epoch_{epoch}"))

    model.save_pretrained(os.path.join(checkpoint_dir, "final"))

shortened_precision = {
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    seed(args)

    wandb_group = args.wandb_group if args.wandb_group is not None else args.root_dir.split("/")[-1]
    wandb_name = args.wandb_name if args.wandb_name is not None else wandb_group
    accelerator = Accelerator(mixed_precision=shortened_precision[args.precision],
                              log_with="wandb" if args.use_wandb else None)
    
    if args.use_wandb:
        accelerator.init_trackers("dial", init_kwargs={
                                "wandb": {
                                    "group": wandb_group,
                                    "name": wandb_name,
                                    "config": vars(args)
                                }
                              })

    model, optimizer, dataloader, loss_fn, eval_dataloader, eval_loss_fn = initialize(args, accelerator)
    train(model, optimizer, dataloader, loss_fn, args, accelerator, eval_dataloader, eval_loss_fn)

if __name__ == "__main__":
    main()
