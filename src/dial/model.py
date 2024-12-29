from typing import Optional, List
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import os
import json
from peft import get_peft_model, LoraConfig

from dial.multihead_model import PeftModelForDualHeads
from dial.arguments import Arguments
from dial.utils import (
    to_device,
    init_class,
    str_to_task_type,
    get_target_modules,
    select_last_token_hidden_state,
    str_to_dtype,
    mark_lora_layers_as_trainable
)
from da import DABase

class Model(nn.Module):
    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        extra_args = {"num_labels": args.num_labels} if args.model_type == "sequence_classification" else {}
        self.model = init_class[args.model_type].from_pretrained(
            args.model_name,
            torch_dtype=str_to_dtype[args.precision],
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            **extra_args,
        )

        self.hidden_size = self.model.config.hidden_size if args.model_type != "embedding" else self.model.config.text_config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True
        )
        self.right_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            padding_side="right",
            truncation_side="right",
            trust_remote_code=True
        )
        
        if self.args.add_pad_token:
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
            if self.right_tokenizer.pad_token_id is None:
                self.right_tokenizer.add_special_tokens({"pad_token": "<pad>"})
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            assert self.model.config.pad_token_id == self.tokenizer.pad_token_id
            assert self.model.config.pad_token_id == self.right_tokenizer.pad_token_id

        if self.args.add_cls_token:
            if self.tokenizer.cls_token_id is None:
                self.tokenizer.add_special_tokens({"cls_token": "<cls>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
            if self.right_tokenizer.cls_token_id is None:
                self.right_tokenizer.add_special_tokens({"cls_token": "<cls>"})
            if self.model.config.cls_token_id is None:
                self.model.config.cls_token_id = self.tokenizer.cls_token_id
            
            assert self.model.config.cls_token_id == self.tokenizer.cls_token_id
            assert self.model.config.cls_token_id == self.right_tokenizer.cls_token_id

        if self.args.adapter_path is not None:
            self.model.load_adapter(self.args.adapter_path)
            mark_lora_layers_as_trainable(self.model)
        elif self.args.lora_r is not None:
            peft_config = LoraConfig(
                task_type=str_to_task_type[args.model_type],
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout
            )

            if self.args.target_modules == "custom_phrase":
                peft_config.target_modules = get_target_modules(self.model, self.args.target_modules_custom_phrase)
            elif self.args.target_modules is not None:
                peft_config.target_modules = self.args.target_modules

            if self.args.model_type == "gemma_dual":
                self.model = PeftModelForDualHeads(self.model, peft_config, autocast_adapter_dtype=True)
                self.model.base_model.model.lm_head.weight.requires_grad = True
            else:
                self.model = get_peft_model(self.model, peft_config)

            self.model.print_trainable_parameters()
        
        if self.args.model_type in ["causal_lm", "gemma_dual"]:
            self.right_tokenizer.add_bos_token = False

        if self.args.train_mode in ["domain_adaptation", "domain_adaptation_contrastive", "domain_adaptation_mcq"]:
            with open(args.da_config_path, "r") as f:
                da_config = json.load(f)
            self.da_base = DABase(embeds_size=[self.hidden_size], **da_config)
        
        self.mode = "eval"
        self.model_config = self.model.config
        self.global_step = 0

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.right_tokenizer.save_pretrained(save_dir)

    def tokenize(self, text: List[str], side: str = "left", max_length: int = None):
        tokenizer = self.tokenizer if side == "left" else self.right_tokenizer
        if max_length is None:
            if self.mode == "eval" and self.args.eval_max_length is not None:
                max_length = self.args.eval_max_length
            else:
                max_length = self.args.max_length
        return tokenizer(
            text,
            padding="max_length" if self.global_step == 0 else True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def run_ntp(self, texts: List[str], dual_model: bool = False):
        tokenized_texts = self.tokenize(texts, side="left")
        inputs = to_device(tokenized_texts, self.model.device)

        label_mask = inputs["attention_mask"]
        labels = label_mask * inputs["input_ids"] - 100 * (1 - label_mask)
        labels = labels.to(self.model.device)

        label_kwargs = {"causal_labels": labels} if dual_model else {"labels": labels}
        outputs = self.model(**inputs, **label_kwargs)
        return outputs

    def run_sft(self, prompts: List[str], responses: List[str], dual_model: bool = False):
        tokenized_prompts = self.tokenize(prompts, side="left", max_length=self.args.prompt_max_length if self.args.prompt_max_length is not None else self.args.max_length // 2)
        tokenized_responses = self.tokenize(responses, side="right", max_length=self.args.response_max_length if self.args.response_max_length is not None else self.args.max_length // 2)

        inputs = {
            k: torch.cat([tokenized_prompts[k], tokenized_responses[k]], dim=1)
            for k in tokenized_prompts.keys()
        }
        inputs = to_device(inputs, self.model.device)

        label_mask = torch.cat([torch.zeros_like(tokenized_prompts["input_ids"]), tokenized_responses["attention_mask"]], dim=1).to(self.model.device)
        labels = label_mask * inputs["input_ids"] - 100 * (1 - label_mask)
        labels = labels.to(self.model.device)

        label_kwargs = {"causal_labels": labels} if dual_model else {"labels": labels}
        outputs = self.model(**inputs, **label_kwargs)
        return outputs

    def run_class(self, texts: List[str]):
        if self.args.add_cls_token:
            texts = [f"{text}<cls>" for text in texts]
        tokenized = self.tokenize(texts, side="right")
        inputs = to_device(tokenized, self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)

        if self.args.embedding_selection in ["last_token", "cls_token"]:
            embeddings = select_last_token_hidden_state(
                self.model,
                inputs["input_ids"],
                outputs.hidden_states[-1],
                self.model_config.pad_token_id
            )
        elif self.args.embedding_selection == "pooled":
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states * inputs["attention_mask"].unsqueeze(-1)
            embeddings = embeddings.sum(1) / inputs["attention_mask"].sum(1).unsqueeze(-1)

        predictions = outputs.logits if hasattr(outputs, "logits") else outputs.sequence_logits
        return predictions, embeddings