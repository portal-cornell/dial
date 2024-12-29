from typing import Dict
import json
import numpy as np
import torch
from dial.arguments import Arguments
from scipy.stats import spearmanr, pearsonr

def mcq_accuracy_metric(outputs: torch.Tensor, args: Arguments) -> float:
    # outputs: shape (B, accuracy_columns)
    # First column is the correct answer, the rest are the predicted scores
    columns = args.accuracy_columns
    assert outputs.shape[1] == columns

    new_outputs = outputs.argmax(dim=1)
    return (new_outputs == 0).float().mean().item()

def mcq_rank_metric(outputs: torch.Tensor, args: Arguments) -> float:
    # outputs: shape (B, accuracy_columns, 1)
    # First column is the correct answer, the rest are the predicted scores
    columns = args.accuracy_columns
    assert outputs.shape[1] == columns
    new_outputs = outputs.reshape(-1, columns)

    # Calculate the rank of the correct answer (index 0) need to sort the values in descending order
    # e.g. if outputs at index i is [0.1, 0.3, 0.0, 0.4], rank is 2 (0-indexed), as it is the third largest.
    sorted_indices = new_outputs.argsort(descending=True, dim=1)
    ranks = torch.where(sorted_indices == 0)[1]
    return ranks.float().mean().item()

def multiclass_accuracy_metric(outputs: torch.Tensor, args: Arguments) -> float:
    columns = args.accuracy_columns

    pairs = outputs.reshape(-1, columns-1, 2)
    new_outputs = torch.zeros([len(pairs), columns])
    new_outputs[:, 0] = pairs[:, 0, 0]
    new_outputs[:, 1:] = pairs[:, :, 1]
    new_outputs = new_outputs.argmax(dim=1)
    return (new_outputs == 0).float().mean().item()

def average_rank_metric(outputs: torch.Tensor, args: Arguments) -> float:
    columns = args.accuracy_columns

    pairs = outputs.reshape(-1, columns-1, 2)
    new_outputs = torch.zeros([len(pairs), columns])
    new_outputs[:, 0] = pairs[:, 0, 0]
    new_outputs[:, 1:] = pairs[:, :, 1]

    # Calculate the rank of the correct answer (index 0) need to sort the values in descending order
    # e.g. if outputs at index i is [0.1, 0.3, 0.0, 0.4], rank is 2 (0-indexed), as it is the third largest.
    sorted_indices = new_outputs.argsort(descending=True, dim=1)
    ranks = torch.where(sorted_indices == 0)[1]
    return ranks.float().mean().item()

class CorrelationMetric:
    def __init__(self, variant: str):
        self.variant = variant

    def __call__(self, outputs: torch.Tensor, args: Arguments) -> float:
        outputs = outputs.float().cpu().detach().numpy()
        if self.variant == "pearson":
            return pearsonr(outputs[:, 0], outputs[:, 1]).statistic
        elif self.variant == "spearman":
            return spearmanr(outputs[:, 0], outputs[:, 1]).statistic

metric_name_mapping = {
    "multiclass_accuracy": multiclass_accuracy_metric,
    "average_rank": average_rank_metric,
    "pearson": CorrelationMetric("pearson"),
    "spearman": CorrelationMetric("spearman"),
    "mcq_accuracy": mcq_accuracy_metric,
    "mcq_rank": mcq_rank_metric
}

def get_metrics(outputs: torch.Tensor, args: Arguments) -> Dict[str, float]:
    return {
        k: metric_name_mapping[k](outputs, args) for k in args.metric_names
    }