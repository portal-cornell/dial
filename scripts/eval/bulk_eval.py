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

@torch.no_grad()
def bulk_eval(model: Model, dataloader: torch.utils.data.DataLoader, args: Arguments, accelerator: Accelerator, global_step: Optional[int] = None):
    seed(args)
    model.mode = "eval"
    model.eval()
    model = torch.compile(model)
    all_metrics = {}
    pbar = tqdm.tqdm(dataloader)
    all_outputs = []
    all_embeddings = []
    for step, batch in enumerate(pbar):
        with accelerator.autocast():
            inputs, targets = batch
            if len(inputs) == 2:
                prompts, responses = inputs
                inputs = [prompt+response for prompt, response in zip(prompts, responses)]
            elif len(inputs) == 3:
                prompts, responses_chosen, responses_rejected = inputs
                inputs_chosen = [prompt+response for prompt, response in zip(prompts, responses_chosen)]
                inputs_rejected = [prompt+response for prompt, response in zip(prompts, responses_rejected)]
                outputs_chosen, embeddings_chosen = model.run_class(inputs_chosen)
                outputs_rejected, embeddings_rejected = model.run_class(inputs_rejected)
                outputs = torch.cat([outputs_chosen, outputs_rejected], dim=1).reshape(-1, 1)
                embedding_dim = embeddings_chosen.shape[-1]
                embeddings = torch.stack([embeddings_chosen, embeddings_rejected], dim=1).reshape(-1, embedding_dim)
            else:
                inputs = inputs[0]
                outputs, embeddings = model.run_class(inputs)
            all_outputs.append(outputs)
            all_embeddings.append(embeddings)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_outputs, all_embeddings

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    seed(args)

    accelerator = Accelerator()

    if args.use_wandb:
        setup_wandb(args)

    model, optimizer, dataloader, loss_fn, eval_dataloader, eval_loss_fn = initialize(args, accelerator)

    all_outputs, all_embeddings = bulk_eval(model, eval_dataloader, args, accelerator)
    output_path, embedding_path = args.bulk_eval_output_path, args.bulk_eval_embedding_path

    print(f"Saving outputs to {output_path}")
    torch.save(all_outputs.cpu().detach(), output_path)

    print(f"Saving embeddings to {embedding_path}")
    torch.save(all_embeddings.cpu().detach(), embedding_path)

if __name__ == "__main__":
    main()