from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class LLaVaModelHF(object):
    def __init__(self, model_name, device, dtype, cache_dir=None, **kwargs):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        print(f"Cache dir: {cache_dir}")
        use_flash_attention = kwargs.get("use_flash_attn", False)
        attn_implementation = "flash_attention_2" if use_flash_attention else None
        
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=self.dtype, low_cpu_mem_usage=True, 
                                                                       device_map='auto', cache_dir=cache_dir, attn_implementation=attn_implementation)
        self.model.eval()
        print(f"Model loaded on device: {self.model.device}")

        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_beams = kwargs.get("num_beams", 1)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 256)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)

    def generate(self, prompt, return_prompt=False, image_files=None, temperature=None, max_new_tokens=None):
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
            
        if image_files is None and '<image>' in prompt:
            prompt = prompt.replace('<image>', '')
            
        image_path = image_files[0] if type(image_files) == list else image_files
        image = Image.open(image_path) if image_files is not None else None
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_k=self.top_k, top_p=self.top_p, num_beams=self.num_beams, 
        num_return_sequences=self.num_return_sequences, max_new_tokens=max_new_tokens, min_new_tokens=self.min_new_tokens, pad_token_id=self.processor.tokenizer.eos_token_id)
        
        outputs = outputs[0][len(inputs['input_ids'][0]):] if not return_prompt else outputs[0]
        decoded_output = self.processor.tokenizer.decode(outputs, skip_special_tokens=True)
        
        return decoded_output
