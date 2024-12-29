from typing import Callable, Optional
from dial.common import initialize
from dial.arguments import Arguments
from dial.model import Model
from dial.utils import setup_wandb, seed
from dial.metrics import get_metrics
from transformers import HfArgumentParser
from accelerate import Accelerator
import torch
import tqdm
import wandb

@torch.no_grad()
def evaluate(model: Model, dataloader: torch.utils.data.DataLoader, loss_fn: Callable, args: Arguments, accelerator: Accelerator, global_step: Optional[int] = None):
    seed(args)
    model.mode = "eval"
    model.eval()
    all_metrics = {}
    pbar = tqdm.tqdm(dataloader)
    all_outputs = []
    for step, batch in enumerate(pbar):
        with accelerator.autocast():
            loss, extra_metrics, outputs = loss_fn(batch, model, args)
        all_outputs.append(outputs)

        extra_metrics = {f"eval/{k}": v for k, v in extra_metrics.items()}
        curr_metrics = {
            "eval/loss": loss.item(),
            **extra_metrics,
        }

        for k, v in curr_metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

        pbar.set_description(f"Eval step {step}, Loss: {loss.item():.3f}")

    all_outputs = torch.cat(all_outputs, dim=0)
    final_metrics = get_metrics(all_outputs, args)
    final_metrics = {f"eval/{k}": v for k, v in final_metrics.items()}

    mean_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    mean_metrics.update(final_metrics)
    if args.use_wandb:
        wandb.log(mean_metrics, step=global_step)

    return mean_metrics

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    seed(args)

    accelerator = Accelerator()

    if args.use_wandb:
        setup_wandb(args)

    model, optimizer, dataloader, loss_fn, eval_dataloader, eval_loss_fn = initialize(args, accelerator)

    metrics = evaluate(model, eval_dataloader, eval_loss_fn, args, accelerator)
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()