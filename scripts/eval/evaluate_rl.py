from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import sys
import os
from dataclasses import dataclass
from argparse_dataclass import ArgumentParser
from scripts.dataproc.batch_inference import batch_inference
from dial.evaluation_functions import evaluate_xsdata
from vllm import LLM

INVALID_LOGPROB = 1.0

@dataclass
class FullArguments:
    ppo_model_name: str = "google/gemma-1.1-2b-it"
    ppo_model_path: Optional[str] = None

    data_path: Optional[str] = None
    system_message: Optional[str] = None

    address: str = "localhost"
    port: int = 7502
    seed: int = 0

    ppo_response_max_length: int = 320
    ppo_temperature: float = 0.0

def main():
    parser = ArgumentParser(FullArguments)
    args = parser.parse_args(sys.argv[1:])

    print(f"Setting seed to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.data_path is not None and os.path.exists(args.data_path) and args.data_path.endswith(".json")
    with open(args.data_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)

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

    print("Evaluating generated responses")
    print("----------")
    evaluation_fn = evaluate_xsdata
    evaluation_result = evaluation_fn(correct_data, generated_responses, address=args.address, port=args.port)
    print("Evaluation result", evaluation_result)
    print("----------")

def evaluate(
        ppo_model_path: str, 
        data_path: str, 
        llm: Optional[LLM] = None,
        address: str = "localhost", 
        port: int = 7502) -> float:
    seed = 0
    print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert data_path is not None and os.path.exists(data_path) and data_path.endswith(".json")
    with open(data_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)

    prompts = list(set([d["prompt"] for d in data]))
    correct_mapping = {}
    for d in data:
        correct_mapping[d["prompt"]] = d["correct"]
    correct_data = [{"prompt": prompt, "correct": correct_mapping[prompt]} for prompt in prompts]
    generated_responses, full_prompts = batch_inference(
        descriptions=prompts,
        llm=llm,
        model_name="google/gemma-1.1-2b-it",
        model_path=ppo_model_path,
        stream=False,
        do_sample=True,
        max_new_tokens=320,
        temperature=0.0,
        top_p=None,
        top_k=None,
        num_return_sequences=1,
        extra_stop_token=None,
        lora=ppo_model_path is not None,
        gpu_memory_utilization=0.9,
        system_message=None,
        apply_prompt_processor=True
    )
    generated_responses = [response[0] for response in generated_responses]

    print("Evaluating generated responses")
    print("----------")
    evaluation_fn = evaluate_xsdata
    evaluation_result = evaluation_fn(correct_data, generated_responses, address=address, port=port)
    print("Evaluation result", evaluation_result)
    print("----------")
    return evaluation_result

if __name__ == "__main__":
    main()