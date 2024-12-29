from typing import Optional, List

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from fastchat.model.model_adapter import get_conversation_template
import os
import json
import argparse

class PromptProcessor:
    def __init__(self, model_name: str, history: bool = False):
        self.history = history
        if os.path.exists(model_name):
            config_path = os.path.join(model_name, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            model_name = config["_name_or_path"]
            print(f"Loaded from path {config_path}: {model_name}")

        self.conversation = get_conversation_template(model_name)
    
    def process(self, prompt: str):
        if not self.history:
            self.conversation.messages = []
        self.conversation.append_message(self.conversation.roles[0], prompt)
        self.conversation.append_message(self.conversation.roles[1], None)
        return self.conversation.get_prompt()

def get_llm(model_name: str, 
            lora: bool = False, 
            gpu_memory_utilization: float = 0.9, 
            tensor_parallel_size: int = 1):
    llm = LLM(model_name, enable_lora=lora, max_lora_rank=64, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size)
    return llm

def batch_inference(descriptions: List[str], 
                    model_name: str,
                    model_path: Optional[str] = None,
                    llm: Optional[LLM] = None,
                    stream: bool = False,
                    do_sample: bool = False,
                    max_new_tokens: int = 3072,
                    temperature: float = 0.0,
                    top_p: Optional[float] = None,
                    top_k: Optional[int] = None,
                    num_return_sequences: int = 1,
                    extra_stop_token: Optional[int] = None,
                    lora: bool = False,
                    gpu_memory_utilization: float = 0.9,
                    system_message: Optional[str] = None,
                    apply_prompt_processor: bool = True):
    assert not stream, "Stream is not supported in batch inference"
    if apply_prompt_processor:
        prompt_processor = PromptProcessor(model_name)
        if system_message is not None:
            prompt_processor.conversation.system_message = system_message
        prompts = [prompt_processor.process(description) for description in descriptions]
    else:
        print("Prompt processor is disabled")
        prompts = descriptions

    if llm is None:
        llm = get_llm(model_name if (model_path is None or lora) else model_path, lora=lora, gpu_memory_utilization=gpu_memory_utilization)
    lora_request = LoRARequest("lora", 1, model_path) if lora else None
    
    if extra_stop_token is not None:
        stop_token_ids = [extra_stop_token]
    else:
        stop_token_ids = None

    kwargs = {}
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        **kwargs
    )

    request_output = llm.generate(prompts, sampling_params, lora_request=lora_request)
    text_outputs = [[x.text for x in output.outputs] for output in request_output]
    return text_outputs, prompts

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--descriptions_path", type=str, required=True)
    parser.add_argument("--outputs_path", type=str, required=True)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--extra_stop_token", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    parser.add_argument("--system_message", type=str, default=None)
    parser.add_argument("--apply_prompt_processor", type=int, default=True)

    return parser.parse_args()

def main():
    args = get_args()

    assert args.descriptions_path.endswith(".json")
    with open(args.descriptions_path, "r") as f:
        descriptions = json.load(f)
    
    llm = get_llm(args.model_name if (args.model_path is None or args.lora) else args.model_path, 
                  lora=args.lora, 
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  tensor_parallel_size=args.tensor_parallel_size)
    outputs = batch_inference(
        descriptions,
        args.model_name,
        model_path=args.model_path,
        llm=llm,
        stream=args.stream,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_return_sequences,
        extra_stop_token=args.extra_stop_token,
        lora=args.lora,
        gpu_memory_utilization=args.gpu_memory_utilization,
        system_message=args.system_message,
        apply_prompt_processor=args.apply_prompt_processor
    )

    outputs = [[prompt] + response for prompt, response in zip(descriptions, outputs)]

    with open(args.outputs_path, "w") as f:
        json.dump(outputs, f)

    print(f"Outputs saved to {args.outputs_path}")

if __name__ == "__main__":
    main()
