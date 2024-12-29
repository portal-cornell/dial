from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
import math
import time
import itertools
import numpy as np
import wandb
import random
import json
import tqdm
import copy
import sys
import gc
import os
from dataclasses import dataclass
# from transformers import PreTrainedTokenizerBase, AutoTokenizer
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_gemma

apply_liger_kernel_to_gemma()

from peft import LoraConfig, get_peft_model, TaskType
from trl.trainer.utils import first_true_indices
from trl.core import masked_whiten, masked_mean
from dial.utils import select_last_token_hidden_state, to_device, mark_lora_layers_as_trainable
from argparse_dataclass import ArgumentParser
from scripts.batch_inference import batch_inference, PromptProcessor
from dial.evaluation_functions import evaluate_xsdata
from da import DABase

INVALID_LOGPROB = 1.0

@dataclass
class FullArguments:
    train_reward: bool = False
    reward_batch_size: int = 32
    reward_gradient_accumulation_steps: int = 1
    reward_num_epochs: int = 1
    reward_max_length: int = 1024
    reward_learning_rate: float = 5e-5

    train_ppo: bool = False
    ppo_batch_size: int = 32
    ppo_num_epochs: int = 1
    ppo_logprob_batch_size: Optional[int] = None
    ppo_query_max_length: int = 1024
    ppo_response_max_length: int = 1024
    ppo_temperature: float = 1.0
    ppo_missing_eos_penalty: Optional[float] = None
    ppo_kl_coef: float = 0.05
    ppo_whiten_rewards: bool = False
    ppo_gamma: float = 1.0
    ppo_lam: float = 0.95
    ppo_num_optimize_steps: int = 1
    ppo_mini_batch_size: Optional[int] = None
    ppo_cliprange_value: float = 0.2
    ppo_cliprange: float = 0.2
    ppo_vf_coef: float = 0.1
    ppo_gradient_accumulation_steps: int = 1
    ppo_learning_rate: float = 5e-5

    reward_model_name: str = "google/gemma-2b"
    ppo_model_name: str = "google/gemma-1.1-2b-it"
    value_model_name: str = "google/gemma-1.1-2b-it"

    reward_model_path: Optional[str] = None
    ppo_model_path: Optional[str] = None
    value_model_path: Optional[str] = None

    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    data_path: Optional[str] = None
    root_dir: str = ""
    use_wandb: bool = False
    wandb_project: str = "dial"
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    dtype: str = "bfloat16"

    num_generation_epochs: int = 1
    system_message: Optional[str] = None

    reward_da: bool = False
    da_config_path: Optional[str] = None

    address: str = "localhost"
    port: int = 7502
    seed: int = 0

    step_lr: bool = False

str_to_dtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

class RewardModel(nn.Module):
    def __init__(self, model: transformers.AutoModelForSequenceClassification, tokenizer: transformers.PreTrainedTokenizerBase, max_length: int, da_config_path: Optional[str] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_wrapper = SideTokenizerWrapper(tokenizer)
        self.max_length = max_length

        if da_config_path is not None:
            with open(da_config_path, "r") as f:
                da_config = json.load(f)
            self.da_base = DABase(embeds_size=[self.model.config.hidden_size], **da_config)

    def forward(self, inputs, return_dict=True):
        tokenized = self.tokenizer_wrapper("left", inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length, add_bos_token=True)
        inputs = to_device(tokenized, self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)

        embeddings = select_last_token_hidden_state(
            self.model,
            inputs["input_ids"],
            outputs.hidden_states[-1],
            self.tokenizer.pad_token_id
        )
        return outputs.logits, embeddings

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

def reward_train(
        args: FullArguments,
        data: List[Dict[str, str]],
        reward_model: RewardModel,
        reward_optimizer: torch.optim.Optimizer,
        generation_number: int = 0
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = accelerate.Accelerator()

    reward_model, optimizer = accelerator.prepare(reward_model, reward_optimizer)
    if args.reward_da:
        reward_model.da_base.da_net.nets = accelerator.prepare(reward_model.da_base.da_net.nets)

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=args.reward_batch_size,
        shuffle=True,
    )

    reward_model.train()

    for epoch in range(args.reward_num_epochs):
        pbar = tqdm.tqdm(enumerate(dataloader), total=math.ceil(len(data) / args.reward_batch_size))
        for step, batch in pbar:
            prompts = batch["prompt"]
            responses_chosen = batch["response_chosen"]
            responses_rejected = batch["response_rejected"]

            if "target_responses" in batch:
                target_responses = batch["target_responses"][0]
                both_target = [f"{prompt}\n{response}" for prompt, response in zip(prompts, target_responses)]

            both_chosen = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses_chosen)]
            both_rejected = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses_rejected)]

            metrics = {}
            total_loss = torch.tensor(0.0).to(device)
            with accelerator.autocast():
                rewards_chosen, embeddings_chosen = reward_model(both_chosen)
                rewards_rejected, embeddings_rejected = reward_model(both_rejected)

                if args.reward_da and "target_responses" in batch:
                    rewards_target, embeddings_target = reward_model(both_target)
                    source_embeddings = torch.cat((embeddings_chosen, embeddings_rejected), dim=0)
                    all_embeddings = torch.cat((source_embeddings, embeddings_target), dim=0)

                    domain_labels = torch.cat([torch.zeros(len(source_embeddings)), torch.ones(len(embeddings_target))]).to(all_embeddings.device)
                    da_loss, additional_losses = reward_model.da_base.get_da_loss([all_embeddings], domain_labels)
                    total_loss += da_loss
                    for key, value in additional_losses.items():
                        metrics[f"train/{key}"] = value
                    metrics[f"train/da_loss"] = da_loss.item()

                loss = -F.logsigmoid((rewards_chosen - rewards_rejected)).mean()
                total_loss += loss

            total_loss.backward()
            if step % args.reward_gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if args.use_wandb:
                wandb.log({"train/loss": total_loss.item(), "train/step": step, "train/epoch": epoch, **metrics})
            pbar.set_description(f"Epoch {epoch}, step: {step}, loss: {loss.item():.3f}")
        
        save_path = f"{args.root_dir}/reward_model_gen_{generation_number}_{epoch}"
        reward_model.model.save_pretrained(save_path)

    optimizer.zero_grad()
    reward_model.save_pretrained(f"{args.root_dir}/reward_model_gen_{generation_number}_final")
    reward_model.cpu()

class SideTokenizerWrapper:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
    
    def __call__(self, side: str, *args, add_bos_token: Optional[bool] = None, **kwargs):
        assert side in ["left", "right"]
        if side == "right":
            self.tokenizer.padding_side = "right"
            if add_bos_token is None:
                self.tokenizer.add_bos_token = False
        elif side == "left":
            self.tokenizer.padding_side = "left"
            if add_bos_token is None:
                self.tokenizer.add_bos_token = True
            
        if add_bos_token is not None:
            self.tokenizer.add_bos_token = add_bos_token
        return self.tokenizer(*args, **kwargs)

def ppo_train(
        args: FullArguments,
        queries: List[str], 
        generated_responses: List[str],
        generation_number: int,
        ppo_model: transformers.AutoModelForCausalLM,
        value_model: transformers.AutoModelForSequenceClassification,
        reward_model: transformers.AutoModelForSequenceClassification,
        ref_policy: transformers.AutoModelForCausalLM,
        ppo_tokenizer: transformers.PreTrainedTokenizerBase,
        value_tokenizer: transformers.PreTrainedTokenizerBase,
        ppo_optimizer: torch.optim.Optimizer,
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = accelerate.Accelerator()
    tokenizer_wrapper = SideTokenizerWrapper(ppo_tokenizer)

    dataloader = torch.utils.data.DataLoader(
        list(zip(queries, generated_responses)),
        batch_size=args.ppo_batch_size,
        shuffle=True,
    )

    logprob_batch_size = args.ppo_logprob_batch_size if args.ppo_logprob_batch_size is not None else args.ppo_batch_size

    ppo_model, value_model, reward_model, ref_policy = accelerator.prepare(ppo_model, value_model, reward_model, ref_policy)

    print("training policy")
    start_time = time.time()
    ppo_model.train()
    reward_model.eval()
    value_model.train()
    ref_policy.eval()

    mini_batch_size = args.ppo_mini_batch_size if args.ppo_mini_batch_size is not None else args.ppo_batch_size

    for epoch in range(args.ppo_num_epochs):
        pbar = tqdm.tqdm(enumerate(dataloader), total=math.ceil(len(queries) / args.ppo_batch_size))
        for step, batch in pbar:
            batch_queries, batch_responses = batch
            nontokenized_both = [f"{query}\n{response}" for query, response in zip(batch_queries, batch_responses)]
            tokenized_queries = tokenizer_wrapper("left", batch_queries, return_tensors="pt", padding="max_length", truncation=True, max_length=args.ppo_query_max_length)["input_ids"].to(device)
            tokenized_responses = tokenizer_wrapper("right", batch_responses, return_tensors="pt", padding="max_length", truncation=True, max_length=args.ppo_response_max_length)["input_ids"].to(device)
            tokenized_both = torch.cat((tokenized_queries, tokenized_responses), dim=1)
            with torch.no_grad(), accelerator.autocast():
                context_length = tokenized_queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []

                for i in range(0, tokenized_queries.shape[0], logprob_batch_size):
                    curr_queries = tokenized_queries[i : i + logprob_batch_size]
                    curr_responses = tokenized_responses[i : i + logprob_batch_size]
                    sequence_length = first_true_indices(curr_responses == ppo_tokenizer.pad_token_id) - 1
                    curr_both = torch.cat((curr_queries, curr_responses), dim=1)

                    present_output = ppo_model(curr_both, return_dict=True)
                    logits = present_output.logits[:, context_length - 1 : -1]
                    logits /= args.ppo_temperature + 1e-7
                    all_logprobs = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprobs, 2, curr_responses.unsqueeze(-1)).squeeze(-1)
                    del present_output, logits, all_logprobs

                    ref_output = ref_policy(curr_both, return_dict=True)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.ppo_temperature + 1e-7
                    ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprobs, 2, curr_responses.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprobs

                    full_value = value_model.score(value_model(curr_both, return_dict=True, output_hidden_states=True).hidden_states[-1])
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    score, _ = reward_model(nontokenized_both, return_dict=True)

                    responses.append(curr_responses)
                    postprocessed_responses.append(curr_responses)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score.flatten())
                    values.append(value)

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score)

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == ppo_tokenizer.eos_token_id, dim=-1)
                if args.ppo_missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= args.ppo_missing_eos_penalty

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.ppo_kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.ppo_whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.ppo_gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.ppo_gamma * args.ppo_lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)

            curr_metrics = {
                "approxkl": [],
                "pg_clipfrac": [],
                "pg_loss": [],
                "vf_loss": [],
                "vf_clipfrac": [],
                "entropy": [],
                "ratio": [],
            }
            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            batch_global_step = 0
            for optimize_step in range(args.ppo_num_optimize_steps):
                for start_index in range(0, tokenized_queries.shape[0], mini_batch_size):
                    mb_advantage = advantages[start_index : start_index + mini_batch_size]
                    mb_responses = responses[start_index : start_index + mini_batch_size]
                    mb_both = tokenized_both[start_index : start_index + mini_batch_size]
                    mb_logprobs = logprobs[start_index : start_index + mini_batch_size]
                    mb_return = returns[start_index : start_index + mini_batch_size]
                    mb_values = values[start_index : start_index + mini_batch_size]

                    with accelerator.autocast():
                        output = ppo_model(mb_both, return_dict=True)
                        vpred_temp = value_model.score(value_model(mb_both, return_dict=True, output_hidden_states=True).hidden_states[-1])
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.ppo_temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(
                            new_logprobs, padding_mask[start_index : start_index + mini_batch_size], INVALID_LOGPROB
                        )
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, padding_mask_p1[start_index : start_index + mini_batch_size], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.ppo_cliprange_value,
                            mb_values + args.ppo_cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[start_index : start_index + mini_batch_size])
                        vf_clipfrac = masked_mean(
                            (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[start_index : start_index + mini_batch_size]
                        )
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.ppo_cliprange, 1.0 + args.ppo_cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~padding_mask[start_index : start_index + mini_batch_size])
                        loss = pg_loss + args.ppo_vf_coef * vf_loss
                    loss.backward()
                    if batch_global_step % args.ppo_gradient_accumulation_steps == 0:
                        ppo_optimizer.step()
                        ppo_optimizer.zero_grad()
                    batch_global_step += 1
                    with torch.no_grad():
                        pg_clipfrac = masked_mean(
                            (pg_losses2 > pg_losses).float(), ~padding_mask[start_index : start_index + mini_batch_size]
                        )
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        curr_metrics["approxkl"].append(approxkl.item())
                        curr_metrics["pg_clipfrac"].append(pg_clipfrac.item())
                        curr_metrics["pg_loss"].append(pg_loss.item())
                        curr_metrics["vf_loss"].append(vf_loss.item())
                        curr_metrics["vf_clipfrac"].append(vf_clipfrac.item())
                        curr_metrics["entropy"].append(entropy.mean().item())
                        curr_metrics["ratio"].append(ratio.mean().item())
                        # del everything and empty cache
                        # fmt: off
                        del (
                            output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                            vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                            pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                            mb_advantage, mb_values, mb_responses, mb_both, mb_logprobs,
                        )
                        # fmt: on
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                mean_pg_loss = np.mean(curr_metrics["pg_loss"])
                mean_vf_loss = np.mean(curr_metrics["vf_loss"])
                metrics = {
                    "objective/kl": mean_kl.item(),
                    "objective/entropy": mean_entropy.item(),
                    "objective/non_score_reward": mean_non_score_reward.item(),
                    "objective/rlhf_reward": rlhf_reward.item(),
                    "objective/scores": scores.mean().item(),
                    "policy/approxkl_avg": np.mean(curr_metrics["approxkl"]),
                    "policy/clipfrac_avg": np.mean(curr_metrics["pg_clipfrac"]),
                    "loss/policy_avg": np.mean(curr_metrics["pg_loss"]),
                    "loss/value_avg": np.mean(curr_metrics["vf_loss"]),
                    "val/clipfrac_avg": np.mean(curr_metrics["vf_clipfrac"]),
                    "policy/entropy_avg": np.mean(curr_metrics["entropy"]),
                    "val/ratio": np.mean(curr_metrics["ratio"]),
                    "val/ratio_var": np.var(curr_metrics["ratio"]),
                    "val/num_eos_tokens": (responses == ppo_tokenizer.eos_token_id).sum().item(),
                    "lr": ppo_optimizer.param_groups[0]["lr"],
                    "episode": epoch,
                }

                if args.use_wandb:
                    wandb.log(metrics)
                
                pbar.set_description(f"Epoch {epoch}, step: {step}, pg_loss: {mean_pg_loss:.3f}, vf_loss: {mean_vf_loss:.3f}, kl: {mean_kl:.3f}, non_score_reward: {mean_non_score_reward:.3f}, rlhf_reward: {rlhf_reward:.3f}")

            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            del (
                tokenized_both,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()
            import gc; gc.collect()
        ppo_model.save_pretrained(f"{args.root_dir}/ppo_model_gen_{generation_number}_epoch_{epoch}")
        value_model.save_pretrained(f"{args.root_dir}/value_model_gen_{generation_number}_epoch_{epoch}")

        ppo_tokenizer.save_pretrained(f"{args.root_dir}/ppo_tokenizer_gen_{generation_number}_epoch_{epoch}")
        value_tokenizer.save_pretrained(f"{args.root_dir}/value_tokenizer_gen_{generation_number}_epoch_{epoch}")

    ppo_model.save_pretrained(f"{args.root_dir}/ppo_model_gen_{generation_number}_final")
    value_model.save_pretrained(f"{args.root_dir}/value_model_gen_{generation_number}_final")

    ppo_tokenizer.save_pretrained(f"{args.root_dir}/ppo_tokenizer_gen_{generation_number}_final")
    value_tokenizer.save_pretrained(f"{args.root_dir}/value_tokenizer_gen_{generation_number}_final")

    ppo_model.cpu(); value_model.cpu(); reward_model.cpu(); ref_policy.cpu()

    import gc; gc.collect()
    torch.cuda.empty_cache()

def main():
    # Train reward model

    # Iterate between the following:
    # 1. Sample responses from the policy model
    # 1. Train policy model
    # 2. Evaluate policy model
    # 3. Train reward model with domain adaptation
    # 4. Evaluate reward model

    parser = ArgumentParser(FullArguments)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = accelerate.Accelerator()

    print(f"Setting seed to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Initializing reward model from {args.reward_model_name}")
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(args.reward_model_name)
    reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name, 
        device_map="cpu", 
        torch_dtype=str_to_dtype[args.dtype],
        num_labels=1,
        attn_implementation="flash_attention_2",
    )
    if args.reward_model_path is not None:
        print(f"Loading reward model from {args.reward_model_path}")
        reward_model.load_adapter(args.reward_model_path)
        mark_lora_layers_as_trainable(reward_model)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        reward_model = get_peft_model(reward_model, lora_config)
        reward_model.print_trainable_parameters()

    reward_model = RewardModel(reward_model, reward_tokenizer, args.reward_max_length, args.da_config_path)
    if args.train_reward:
        reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=args.reward_learning_rate)
    # reward_model = accelerator.prepare(reward_model)

    if args.train_ppo:
        print(f"Initializing PPO model from {args.ppo_model_name}")
        ppo_tokenizer = transformers.AutoTokenizer.from_pretrained(args.ppo_model_name)
        ppo_model = transformers.AutoModelForCausalLM.from_pretrained(args.ppo_model_name, device_map="cpu", torch_dtype=str_to_dtype[args.dtype])
        ref_policy = transformers.AutoModelForCausalLM.from_pretrained(args.ppo_model_name, device_map="cpu", torch_dtype=str_to_dtype[args.dtype])
        for param in ref_policy.parameters():
            param.requires_grad = False
        if args.ppo_model_path is not None:
            print(f"Loading PPO model from {args.ppo_model_path}")
            ppo_model.load_adapter(args.ppo_model_path)
            mark_lora_layers_as_trainable(ppo_model)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            ppo_model = get_peft_model(ppo_model, lora_config)
            ppo_model.print_trainable_parameters()

        print(f"Initializing value model from {args.value_model_name}")
        value_tokenizer = transformers.AutoTokenizer.from_pretrained(args.value_model_name)
        value_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                args.value_model_name, 
                device_map="cpu", 
                torch_dtype=str_to_dtype[args.dtype],
                num_labels=1,
                attn_implementation="flash_attention_2",
            )
        if args.value_model_path is not None:
            print(f"Loading value model from {args.value_model_path}")
            value_model.load_adapter(args.value_model_path)
            mark_lora_layers_as_trainable(value_model)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            value_model = get_peft_model(value_model, lora_config)
            value_model.print_trainable_parameters()
        ppo_optimizer = torch.optim.AdamW(itertools.chain(ppo_model.parameters(), value_model.parameters()), lr=args.ppo_learning_rate)
        # ppo_model, value_model, ppo_optimizer = accelerator.prepare(ppo_model, value_model, ppo_optimizer)

        ppo_model.cpu(); value_model.cpu(); ref_policy.cpu()

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    assert args.data_path is not None and os.path.exists(args.data_path) and args.data_path.endswith(".json")
    with open(args.data_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)

    if args.use_wandb:
        group_name = args.wandb_group if args.wandb_group is not None else args.root_dir.split("/")[-1]
        wandb.init(
            project=args.wandb_project,
            group=group_name,
            name=args.wandb_name if args.wandb_name is not None else group_name,
        )

    # Train reward model
    if args.train_reward:
        print("Training reward model")
        prompt_processor = PromptProcessor(args.reward_model_name)
        if args.system_message is not None:
            prompt_processor.conversation.system_message = args.system_message
        prompts = [prompt_processor.process(x["prompt"]) for x in data]
        processed_data = copy.deepcopy(data)
        for i in range(len(data)):
            processed_data[i]["prompt"] = prompts[i]
        reward_data = [{"prompt": x["prompt"], "response_chosen": x["response_chosen"], "response_rejected": x["response_rejected"]} for x in processed_data]
        reward_train(args, reward_data, reward_model, reward_optimizer, generation_number="pre")


    all_generated_responses = {}
    for generation_number in range(args.num_generation_epochs):
        import gc; gc.collect()
        torch.cuda.empty_cache()

        if generation_number == 0:
            args.reward_model_path = f"{args.root_dir}/reward_model_gen_pre_final"
        elif args.train_reward and args.reward_da:
            args.reward_model_path = f"{args.root_dir}/reward_model_gen_{generation_number - 1}_final"
        if generation_number > 0:
            args.ppo_model_path = f"{args.root_dir}/ppo_model_gen_{generation_number - 1}_final"
            args.value_model_path = f"{args.root_dir}/value_model_gen_{generation_number - 1}_final"

        prompts = list(set([d["prompt"] for d in data]))
        correct_mapping = {}
        for d in data:
            correct_mapping[d["prompt"]] = d["correct"]
        correct_data = [{"prompt": prompt, "correct": correct_mapping[prompt]} for prompt in prompts]
        generated_responses, full_prompts = batch_inference(
            descriptions=prompts,
            model_name=args.ppo_model_name,
            model_path=args.ppo_model_path,
            stream=False,
            do_sample=True,
            max_new_tokens=args.ppo_response_max_length,
            temperature=args.ppo_temperature,
            top_p=None,
            top_k=None,
            num_return_sequences=1,
            extra_stop_token=None,
            lora=args.ppo_model_path is not None,
            gpu_memory_utilization=0.9,
            system_message=args.system_message,
            apply_prompt_processor=True
        )
        generated_responses = [response[0] for response in generated_responses]
        for prompt, response in zip(prompts, generated_responses):
            all_generated_responses[prompt] = [response]

        print("Evaluating generated responses")
        print("----------")
        print("Epoch", generation_number)
        evaluation_fn = evaluate_xsdata
        evaluation_result = evaluation_fn(correct_data, generated_responses, address=args.address, port=args.port)
        if args.use_wandb:
            wandb.log({
                "eval/accuracy": evaluation_result
            })
        print("Evaluation result", evaluation_result)
        print("----------")

        import gc; gc.collect()
        torch.cuda.empty_cache()

        if args.train_reward and args.reward_da:
            print("Training reward model with domain adaptation")
            new_data = copy.deepcopy(processed_data)
            for i in range(len(data)):
                new_data[i]["target_responses"] = all_generated_responses[data[i]["prompt"]]
            reward_data = [{"prompt": x["prompt"], "response_chosen": x["response_chosen"], "response_rejected": x["response_rejected"], "target_responses": x["target_responses"]} for x in new_data]
            reward_train(args, reward_data, reward_model, reward_optimizer, generation_number)
            if generation_number % 5 == 4:
                print("Reducing learning rate for reward model")
                for g in reward_optimizer.param_groups:
                    g["lr"] *= 0.5

        import gc; gc.collect()
        torch.cuda.empty_cache()

        # Train PPO model
        if args.train_ppo:
            print("Training PPO model")
            ppo_train(args, 
                      full_prompts, 
                      generated_responses, 
                      generation_number,
                      ppo_model,
                      value_model,
                      reward_model,
                      ref_policy,
                      ppo_tokenizer,
                      value_tokenizer,
                      ppo_optimizer)
            if generation_number % 5 == 4 and args.step_lr:
                print("Reducing learning rate for PPO model")
                for g in ppo_optimizer.param_groups:
                    g["lr"] *= 0.5

if __name__ == "__main__":
    main()