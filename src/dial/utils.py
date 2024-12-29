from typing import Optional, List, Mapping, Iterable, Tuple, Callable
from colorama import Fore
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)
from dial.multihead_model import GemmaForDualHeads
from peft import TaskType
from peft.tuners.lora import LoraLayer
from dial.arguments import Arguments
import wandb
import random
import math

def seed(args: Arguments):
    seed_num = args.seed if args.seed is not None else 0
    print(f"Setting seed to {seed_num}")
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)

def bold(text):
    return f'\033[1m{text}\033[0m'

color_mapping = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "black": Fore.BLACK,
}

def color(text, color):
    return f'{color_mapping[color]}{text}{Fore.RESET}'

def to_device(x, device: str):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return [to_device(v, device) for v in x]
    else:
        return x
    

init_class = {
    "sequence_classification": AutoModelForSequenceClassification,
    "causal_lm": AutoModelForCausalLM,
    "embedding": AutoModel,
    "gemma_dual": GemmaForDualHeads,
}

str_to_task_type = {
    "sequence_classification": TaskType.SEQ_CLS,
    "causal_lm": TaskType.CAUSAL_LM,
    "embedding": TaskType.FEATURE_EXTRACTION,
    "gemma_dual": TaskType.CAUSAL_LM,
}

# Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/tuners_utils.py#L770
def get_target_modules(model: nn.Module, custom_phrase: Optional[str] = None) -> List[str]:
    linear_classes = (nn.Linear, nn.Conv1d)

    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes) and (custom_phrase is None or custom_phrase in name):
            names = name.rsplit(".", 1)[-1]  # get the base name
            linear_module_names.add(names)

    # ignore the last classification head for text generation models
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]

        if last_module_name in linear_module_names:
           linear_module_names -= {last_module_name}
    
    return list(linear_module_names)

def select_last_token_hidden_state(model: nn.Module, input_ids: torch.Tensor, hidden_states: torch.Tensor, pad_token_id: Optional[int] = None):
    batch_size = hidden_states.shape[0]
    if pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(hidden_states.device)
        else:
            sequence_lengths = -1
    
    return hidden_states[torch.arange(batch_size), sequence_lengths]

str_to_dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def setup_wandb(args: Arguments):
    wandb_group = args.wandb_group if args.wandb_group is not None else args.root_dir.split("/")[-1]
    wandb_name = args.wandb_name if args.wandb_name is not None else wandb_group
    wandb.init(
        project="dial",
        group=wandb_group,
        name=wandb_name,
        config=args.__dict__
    )

def is_classifier_weight(name: str) -> bool:
    classifier_terms = ["score", "lm_head", "classifier"]
    return any(term in name for term in classifier_terms)

def mark_lora_layers_as_trainable(model: nn.Module) -> None:
    prefix = "lora_"
    for n, p in model.named_parameters():
        if prefix in n or is_classifier_weight(n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    for active_adapter in model.peft_config:
        bias = model.peft_config[active_adapter].bias
        if bias == "none":
            continue

        if bias == "all":
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in model.modules():
                if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    return model

class PairedDataloader:
    def __init__(self, dataset1, dataset2, batch_size1, batch_size2, shuffle=True):
        self.dataset1, self.dataset2 = dataset1, dataset2
        self.batch_size1, self.batch_size2 = batch_size1, batch_size2
        self.length1, self.length2 = math.ceil(len(self.dataset1) / self.batch_size1), math.ceil(len(self.dataset2) / self.batch_size2)

        self.dataset_indices1 = list(range(len(self.dataset1)))
        self.dataset_indices2 = list(range(len(self.dataset2)))
        self.shuffle = shuffle
        self.shuffle_all()

        self.total_length = max(self.length1, self.length2)
    
    def shuffle_all(self):
        if self.shuffle:
            random.shuffle(self.dataset_indices1)
            random.shuffle(self.dataset_indices2)
        self.index1, self.index2 = 0, 0        

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx >= self.total_length:
            raise IndexError("Index out of bounds.")
        batch1 = []
        for _ in range(self.batch_size1):
            batch1.append(self.dataset1[self.dataset_indices1[self.index1]])
            self.index1 += 1
            if self.index1 >= len(self.dataset1):
                self.index1 = 0
                if self.shuffle:
                    random.shuffle(self.dataset_indices1)
                if idx == self.total_length - 1:
                    break

        batch2 = []
        for _ in range(self.batch_size2):
            batch2.append(self.dataset2[self.dataset_indices2[self.index2]])
            self.index2 += 1
            if self.index2 >= len(self.dataset2):
                self.index2 = 0
                if self.shuffle:
                    random.shuffle(self.dataset_indices2)
                if idx == self.total_length - 1:
                    break

        collated1, collated2 = torch.utils.data.default_collate(batch1), torch.utils.data.default_collate(batch2)
        return collated1[0], collated1[1], collated2[0], collated2[1]