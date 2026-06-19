import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMInferencer:
    def __init__(self, model_path, gpu_ids="0,1,2,3"):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate_answer(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False) 
        
        response = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]
        return response.strip()