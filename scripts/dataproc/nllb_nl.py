'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
'''

from vllm import LLM, SamplingParams
from vllm.inputs import (TokensPrompt, zip_enc_dec_prompts)
import json
import argparse

dtype = "bfloat16"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="eng_Latn")
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    return parser.parse_args()

def main():
    args = get_args()

    # Create a BART encoder/decoder model instance
    llm = LLM(
        model="facebook/nllb-200-3.3B",
        dtype=dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Get BART tokenizer
    tokenizer = llm.get_tokenizer()

    tokenizer.src_lang = args.src_lang
    tgt_lang_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    with open(args.prompts, "r") as f:
        input_prompts = json.load(f)

    split_prompts = []
    split_prompt_mapping = {}
    index_mapping = {}
    for input_prompt in input_prompts:
        index_mapping[input_prompt] = []
        sub_prompts = input_prompt.split("\n")
        for sub_prompt in sub_prompts:
            if sub_prompt in split_prompt_mapping:
                index_mapping[input_prompt].append(split_prompt_mapping[sub_prompt])
            else:
                index_mapping[input_prompt].append(len(split_prompts))
                split_prompt_mapping[sub_prompt] = len(split_prompts)
                split_prompts.append(sub_prompt)

    encoder_prompts, decoder_prompts = [], []
    for input_prompt in split_prompts:
        tokens_prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(
            text=input_prompt, max_length=1024, truncation=True
        ))

        tokens_decoder_prompt = TokensPrompt(prompt_token_ids=[2, tgt_lang_id])

        encoder_prompts.append(tokens_prompt)
        decoder_prompts.append(tokens_decoder_prompt)

    prompts = zip_enc_dec_prompts(
        encoder_prompts,
        decoder_prompts,
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        min_tokens=0,
        max_tokens=1024,
        repetition_penalty=1.15,
    )

    # Generate output tokens from the prompts. The output is a list of
    # RequestOutput objects that contain the prompt, generated
    # text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    final_mapping = {}
    for input_prompt in input_prompts:
        final_mapping[input_prompt] = "\n".join([outputs[index].outputs[0].text for index in index_mapping[input_prompt]])

    with open(args.output_path, "w") as f:
        json.dump(final_mapping, f)

if __name__ == "__main__":
    main()