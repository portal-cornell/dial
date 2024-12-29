from typing import Optional
import json
import torch
from datasets import load_dataset
from dial.arguments import Arguments
from dial.utils import PairedDataloader

class SequencePredictionDataset(torch.utils.data.Dataset):
    def __init__(self, args: Arguments, mode: str, custom_data_files: Optional[str] = None):
        super().__init__()
        self.file_type = args.file_type if mode == "train" or args.eval_file_type is None else args.eval_file_type
        self.data_files = args.data_files if mode == "train" or args.eval_data_files is None else args.eval_data_files
        if custom_data_files is not None:
            self.data_files = custom_data_files

        self.dataset = load_dataset(self.file_type, data_files=self.data_files)["train"]

        dataset_config_path = args.dataset_config_path if mode == "train" or args.eval_dataset_config_path is None else args.eval_dataset_config_path
        with open(dataset_config_path, "r") as f:
            self.dataset_config = json.load(f)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_columns = self.dataset_config["input_columns"]
        target_columns = self.dataset_config["target_columns"]

        inputs = [example[col] for col in input_columns]
        targets = torch.tensor([example[col] for col in target_columns]) if len(target_columns) > 0 else torch.tensor(0.0)
        return inputs, targets

class DomainAdaptationDataset(torch.utils.data.Dataset):
    def __init__(self, args: Arguments, mode: str):
        super().__init__()
        self.data_arguments = args
        self.file_type = args.file_type if mode == "train" or args.eval_file_type is None else args.eval_file_type
        self.data_files = args.data_files if mode == "train" or args.eval_data_files is None else args.eval_data_files

        self.source_dataset = load_dataset(self.file_type, data_files=self.data_files)["train"]
        self.target_dataset = load_dataset(self.file_type, data_files=args.target_data_files)["train"]

        if args.shuffle_target_dataset:
            self.shuffle()

        dataset_config_path = args.dataset_config_path if mode == "train" or args.eval_dataset_config_path is None else args.eval_dataset_config_path
        with open(dataset_config_path, "r") as f:
            self.dataset_config = json.load(f)
    
    def shuffle(self):
        self.random_index_mapping = torch.randperm(len(self.target_dataset))

    def __len__(self):
        return min(len(self.source_dataset), len(self.target_dataset))

    def __getitem__(self, idx):
        source_example = self.source_dataset[idx]
        target_idx = self.random_index_mapping[idx].item() if self.data_arguments.shuffle_target_dataset else idx
        target_example = self.target_dataset[target_idx]

        source_input_columns = self.dataset_config["input_columns"]
        source_target_columns = self.dataset_config["target_columns"]   

        target_input_columns = self.dataset_config["target_input_columns"]
        target_target_columns = self.dataset_config["target_target_columns"]

        source_inputs = [source_example[col] for col in source_input_columns]
        source_targets = torch.tensor([source_example[col] for col in source_target_columns])

        target_inputs = [target_example[col] for col in target_input_columns]
        target_targets = torch.tensor([target_example[col] for col in target_target_columns])

        return source_inputs, source_targets, target_inputs, target_targets

class DomainAdaptationWrappedDataset(torch.utils.data.Dataset):
    def __init__(self, args: Arguments, mode: str):
        super().__init__()
        assert mode == "train", "Domain adaptation dataset only used for training."
        self.data_arguments = args

        self.source_dataset = SequencePredictionDataset(args, mode)
        self.target_dataset = SequencePredictionDataset(args, mode, custom_data_files=args.target_data_files)

        target_batch_size = args.target_batch_size if args.target_batch_size is not None else args.batch_size
        self.dataloader = PairedDataloader(
            self.source_dataset, self.target_dataset, args.batch_size, target_batch_size, shuffle=True
        )

    def __len__(self):
        return len(self.dataloader)
    
    def __getitem__(self, idx):
        return self.dataloader[idx]


str_to_dataset_mapping = {
    "sequence_prediction": SequencePredictionDataset,
    "domain_adaptation": DomainAdaptationDataset,
    "contrastive": SequencePredictionDataset,
    "domain_adaptation_contrastive": DomainAdaptationDataset,
    "domain_adaptation_wrapped": DomainAdaptationWrappedDataset
}

def get_dataset(args: Arguments, mode: str):
    dataset_type = args.dataset_type if mode == "train" or args.eval_dataset_type is None else args.eval_dataset_type
    return str_to_dataset_mapping[dataset_type](args, mode)