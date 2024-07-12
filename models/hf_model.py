import os
import fcntl
import utils

from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFaceModel(object):
    def __init__(self, model_name, device, dtype, cache_dir=None, **kwargs):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        token = os.environ.get("HF_TOKEN", None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir, token=token, device_map='auto')
        self.model.eval()

        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_beams = kwargs.get("num_beams", 1)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 256)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)

        if "tokenizer_pad" in kwargs:
            self.tokenizer.pad_token = kwargs["tokenizer_pad"]

        if "tokenizer_padding_side" in kwargs:
            self.tokenizer.padding_side = kwargs["tokenizer_padding_side"]

    def generate(self, prompt, return_prompt=False, image_files=None, temperature=None, max_new_tokens=None):
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        if '<image>' in prompt:
            prompt = prompt.replace('<image>', '') # Not used for non vision models, this assumes that this class is always used for text models (as the vision model used is LLaVA and is implemented in a different class)
        
        messages = utils.get_messages(prompt)
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_k=self.top_k, top_p=self.top_p, num_beams=self.num_beams, 
        num_return_sequences=self.num_return_sequences, max_new_tokens=max_new_tokens, min_new_tokens=self.min_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        
        outputs = outputs[0][len(inputs[0]):] if not return_prompt else outputs[0]
        decoded_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        # Remove llama special words
        decoded_output = decoded_output.replace("assistant", "").replace("user", "").replace("system", "")

        return decoded_output