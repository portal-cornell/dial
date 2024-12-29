from typing import Tuple, Callable, Optional
import torch
import torch.nn as nn

from dial.arguments import Arguments
from dial.model import Model
from accelerate import Accelerator
from dial.dataset import get_dataset
from dial.loss import get_loss_fn

def initialize(args: Arguments, accelerator: Accelerator) -> Tuple[Model, torch.optim.Optimizer, torch.utils.data.DataLoader, Callable]:
    model = Model(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = get_loss_fn(args, mode="train")
    dataset = get_dataset(args, mode="train")

    if args.dataset_type == "domain_adaptation_wrapped":
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0], shuffle=False)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.mode in ["eval", "both"]:
        eval_dataset = get_dataset(args, mode="eval")
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
        eval_loss_fn = get_loss_fn(args, mode="eval")
    else:
        eval_dataloader = None
        eval_loss_fn = None

    if args.train_mode in ["domain_adaptation", "domain_adaptation_contrastive", "domain_adaptation_mcq"]:
        model.da_base.da_net.nets, model, optimizer, dataloader = accelerator.prepare(model.da_base.da_net.nets, model, optimizer, dataloader)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    return model, optimizer, dataloader, loss_fn, eval_dataloader, eval_loss_fn