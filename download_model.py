import argparse
import os
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    """
    Download a pretrained model from HuggingFace and save it to the hf cache directory.
    """
    parser = argparse.ArgumentParser(description="Download a pretrained model from HuggingFace.")
    parser.add_argument("model_name", type=str, help="The name of the model to download.")
    parser.add_argument("--hf_cache", type=str, default="models/cache/", help="The path to save the model.")
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface auth token.")

    args = parser.parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Downloading model {args.model_name} to {args.hf_cache}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=dtype, cache_dir=args.hf_cache, token=args.hf_token, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, cache_dir=args.hf_cache, 
                                                 token=args.hf_token, local_files_only=False, device_map='auto')
    print(f"Model {args.model_name} downloaded and saved to {args.hf_cache}.")
    
    # print model vocab size
    print(f"Model vocab size: {tokenizer.vocab_size}")

    special_token_dict = tokenizer.special_tokens_map
    tokenizer.add_special_tokens(special_token_dict)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model vocab size after resizing: {tokenizer.vocab_size}")

    print(f"Sample inference:")
    prompt = "Tell me a joke about Symbolic Regression."
    start = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=5000, do_sample=True, temperature=0.9, top_k=50, top_p=0.9, num_return_sequences=1)
    print(f"Prompt: {prompt}")
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = output[len(prompt):]
    end = time.perf_counter()
    print(f"Output: {output}")

    print(f"Model was using device {model.device}")
    print(f"Model took {end-start:.2f} seconds to generate the output.")

if __name__ == "__main__":
    main()