import re
import json
from os import PathLike
import einops
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel, 
    AutoProcessor, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Qwen2ForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
)
from transformers import LlavaNextVideoForConditionalGeneration
from transformers import CLIPConfig, CLIPTokenizer, CLIPModel
from torchvision.transforms.v2 import (
    Compose, 
    Resize, 
    CenterCrop, 
    Lambda, 
    ToTensor, 
    Normalize, 
    ToPILImage,
    functional,
)
from typing import Dict, List, Optional, Union
import os
import math

from abc import ABCMeta, abstractmethod

from care.models.modeling_basemodels import (
    BaseModelForMiniCPMV,
    BaseModelForInternVL2,
    BaseModelForLlavaNextVideo,
    BaseModelForTarsier,
    BaseModelForQwen2VL,
    BaseModelForCaRe,
)
from care.utils.model import load_architectures_from_config
import care.models.qwen_vision_info as qwen_vl_vision_process

captioner_registry = {}

class CaptionMixin(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            captioner_registry[cls.ARCHITECTURE] = cls

    def transform_pixel_values(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        # NOTE: this function doesn't accept unbatched inputs
        # pixel_values should be uint8 of (B, T, C, H, W)
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values)

        if pixel_values.ndim == 4:
            # pixel_values is (B, C, H, W)
            # (B, C, H, W) -> (B, 1, C, H, W)
            pixel_values = pixel_values.unsqueeze(1)
        elif pixel_values.ndim == 5:
            # pixel_values is (B, T, C, H, W)
            pass
        else:
            raise ValueError(f"pixel_values should be 4D or 5D, got {pixel_values.ndim}D")
        return pixel_values

    @abstractmethod
    def describe(self):
        raise NotImplementedError

class AutoCaptioner:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        architecture: Optional[str] = None,
        **kwargs):

        config_path = os.path.join(model_name_or_path, 'config.json')
        if architecture is not None:
            model_arch = architecture
            print(f"Argument `architecture` of AutoEncoder is not None. Overriding model architecture to {model_arch}.")
        else:
            model_arch = load_architectures_from_config(config_path)
        if model_arch not in captioner_registry:
            raise ValueError(
                f"Model architecture {model_arch} is not registered. "
                "You can register it by subclassing EncoderBase and setting ARCHITECTURE attribute."
            )
        if device_map is None:
            if torch.cuda.is_available():
                device_map = 'cuda'
                print(f"Argument `device_map` is None. CUDA is detected. Setting device_map={device_map}.")
            else:
                device_map = 'cpu'
                print(f"Argument `device_map` is None. CUDA is not detected. Setting device_map={device_map}.")
        
        MODEL_CLASS = captioner_registry[model_arch]

        return MODEL_CLASS.from_pretrained(model_name_or_path, load_llm=False, device_map=device_map, **kwargs)
    
class CaptionerForMiniCPMV(BaseModelForMiniCPMV, CaptionMixin):

    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:
        if self.is_llm:
            raise NotImplementedError("describe method is not implemented for LLM models.")
        
        pixel_values = self.transform_pixel_values(pixel_values)

        to_image = ToPILImage()
        batched_frames = []
        for batch in pixel_values:
            frames = [to_image(v) for v in batch]
            batched_frames.append(frames)
        descriptions = []
        for frames in batched_frames:
            msgs = [
                {'role': 'user', 'content': frames + [self.describe_prompt]}
            ]
            params = {}
            params["use_image_id"] = False
            params["max_slice_nums"] = 1
            answer = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                processor=self.processor,
                **params
            )
            descriptions.append(answer)
        return descriptions

class CaptionerForInternVL2(BaseModelForInternVL2, CaptionMixin):

    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:
        
        pixel_values = self.transform_pixel_values(pixel_values)

        dynamic_preprocess_max_num = 1
        prompt = f"<|im_start|>user\n<video>\n{self.describe_prompt}<|im_end|><|im_start|>assistant\n"
        
        self.model.eval()
        IMG_START_TOKEN='<img>'
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        transform = self.build_transform(input_size=448)
        descriptions = []
        
        for batch in pixel_values:
            # batch: (T, C, H, W)
            pixel_values_list, num_patches_list = [], []
            T = batch.shape[0]
            for frame in batch:
                # frame: (C, H, W)
                img = ToPILImage()(frame).convert('RGB')
                img = self.dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=dynamic_preprocess_max_num)
                tiles = [transform(tile) for tile in img]
                tiles = torch.stack(tiles)
                num_patches_list.append(tiles.shape[0])
                pixel_values_list.append(tiles)
            pixel_values = torch.cat(pixel_values_list).to(device=self.model.device, dtype=self.model.dtype)
            if T != 1:
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                prompt = prompt.replace('<video>\n', video_prefix)
            
            generation_config = dict(max_new_tokens=2048, do_sample=True)
            response, _ = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
            descriptions.append(response)
            
        return descriptions

class CaptionerForLlavaNextVideo(BaseModelForLlavaNextVideo, CaptionMixin):
    
    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:
        if self.is_llm:
            raise NotImplementedError("describe method is not implemented for LLM models.")
        
        pixel_values = self.transform_pixel_values(pixel_values)  

        images = None
        videos = None

        if pixel_values.ndim == 4:
            images = list(pixel_values)
        elif pixel_values.ndim == 5:
            videos = list(pixel_values)
            # print(prompt)
        else:
            raise ValueError(f"pixel_values should be 4D or 5D, got {pixel_values.ndim}D")
        prompt = [{
            "role": "user",
            "content": [{"type": "text", "text": f"<video>\n{self.describe_prompt}"}],
        }]
        prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        inputs = self.processor(prompt, images=images, videos=videos, return_tensors="pt").to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.2)

        descriptions = self.processor.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return descriptions

class CaptionerForTarsier(BaseModelForTarsier, CaptionMixin):
    
    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:

        pixel_values = self.transform_pixel_values(pixel_values) # [B, T, C, H, W]
        to_image = ToPILImage()
        batched_frames = []
        for batch in pixel_values:
            frames = [to_image(v) for v in batch]
            batched_frames.append(frames)
        descriptions = []
        generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": 2048,
            "top_p": 1,
            "temperature": 0,
            "use_cache": True
        }

        for frames in batched_frames:
            text_inputs = f"<video>\n{self.describe_prompt}"
            text_inputs = self.processor.process_prompt(text_inputs, frames)
            text_inputs = self.processor.get_text_inputs(text_inputs)
            frames = self.processor.get_pixel_values(frames)
            inputs = {
                "input_ids": text_inputs,
                "pixel_values": frames
            }
            inputs = {k:v.to(self.model.device) for k,v in inputs.items() if v is not None}
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            output_text = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
            descriptions.append(output_text)
        return descriptions

class CaptionerForQwen2VL(BaseModelForQwen2VL, CaptionMixin):
    
    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:
        if self.load_llm:
            raise NotImplementedError("describe method is not implemented for LLM models.")
        
        batched_pixel_values = self.transform_pixel_values(pixel_values)
        descriptions = []
        for pixel_values in batched_pixel_values:
        
            nframes, _, height, width = pixel_values.shape
            min_pixels = qwen_vl_vision_process.VIDEO_MIN_PIXELS
            total_pixels = qwen_vl_vision_process.VIDEO_TOTAL_PIXELS
            max_pixels = max(min(qwen_vl_vision_process.VIDEO_MAX_PIXELS, total_pixels / nframes * qwen_vl_vision_process.FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = 230400
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=qwen_vl_vision_process.IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pixel_values = functional.resize(
                pixel_values,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": f"<video>\n{self.describe_prompt}"}],
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ).replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
            
                    
            inputs = self.processor(
                text=[text],
                images=None,
                videos=[pixel_values],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.2)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            descriptions.append(output_text[0])
        return descriptions

class CaptionerForCaRe(BaseModelForCaRe, CaptionMixin):
    
    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:
        if self.load_llm:
            raise NotImplementedError("describe method is not implemented for LLM models.")
        
        batched_pixel_values = self.transform_pixel_values(pixel_values)
        batched_pixel_values = torch.repeat_interleave(batched_pixel_values, repeats=2, dim=1)
        descriptions = []
        for pixel_values in batched_pixel_values:
            nframes, _, height, width = pixel_values.shape
            min_pixels = qwen_vl_vision_process.VIDEO_MIN_PIXELS
            total_pixels = qwen_vl_vision_process.VIDEO_TOTAL_PIXELS
            max_pixels = max(min(qwen_vl_vision_process.VIDEO_MAX_PIXELS, total_pixels / nframes * qwen_vl_vision_process.FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = 230400
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=qwen_vl_vision_process.IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pixel_values = functional.resize(
                pixel_values,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": f"<video>\n{self.describe_prompt}"}],
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ).replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
            
                    
            inputs = self.processor(
                text=[text],
                images=None,
                videos=[pixel_values],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.2)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            descriptions.append(output_text[0])
        return descriptions