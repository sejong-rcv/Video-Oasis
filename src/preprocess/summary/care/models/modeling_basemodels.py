import os
import math
import torch
import torchvision.transforms as T
import care.models.qwen_vision_info as qwen_vl_vision_process
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlavaNextVideoForConditionalGeneration,
    LlavaConfig, 
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen2VLModel,
    Qwen2VLForConditionalGeneration,
)
from typing import Dict, Optional, Union
from care.models.tarsier.modeling_tarsier import TarsierForConditionalGeneration
from care.models.tarsier.processor import Processor
from care.utils.model import EOL_PROMPTS, load_architectures_from_config
from abc import ABCMeta


base_registry = {}

class BaseModel(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            base_registry[cls.ARCHITECTURE] = cls
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        load_llm: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        **kwargs):
        print(f'Loading {cls.__name__} from {model_name_or_path}')

        return cls(model_name_or_path, load_llm=load_llm, device_map=device_map, **kwargs)

class AutoBase:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        load_llm: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        architecture: Optional[str] = None,
        **kwargs):

        config_path = os.path.join(model_name_or_path, 'config.json')
        if architecture is not None:
            model_arch = architecture
            print(f"Argument `architecture` of AutoBase is not None. Overriding model architecture to {model_arch}.")
        else:
            model_arch = load_architectures_from_config(config_path)
        if model_arch not in base_registry:
            raise ValueError(
                f"Model architecture {model_arch} is not registered. "
                "You can register it by subclassing BaseModel and setting ARCHITECTURE attribute."
            )
        if device_map is None:
            if torch.cuda.is_available():
                device_map = 'cuda'
                print(f"Argument `device_map` is None. CUDA is detected. Setting device_map={device_map}.")
            else:
                device_map = 'cpu'
                print(f"Argument `device_map` is None. CUDA is not detected. Setting device_map={device_map}.")
        
        MODEL_CLASS = base_registry[model_arch]

        return MODEL_CLASS.from_pretrained(model_name_or_path, load_llm=load_llm, device_map=device_map, **kwargs)

class BaseModelForMiniCPMV(BaseModel):

    ARCHITECTURE = "MiniCPMV"
    LLM_CLASS = Qwen2ForCausalLM
    MLLM_CLASS = AutoModel

    @property
    def describe_prompt(self):
        return "Describe the video in detail."
    
    @property
    def text_eol_prompt(self):
        prompt = [{'role': 'user', 'content': EOL_PROMPTS['text']}]
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, use_image_id=False, max_slice_nums=2)
        return prompt
    
    @property
    def image_eol_prompt(self):
        prompt = [{'role': 'user', 'content': EOL_PROMPTS['image']}]
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, use_image_id=False, max_slice_nums=2)
        return prompt
    
    @property
    def video_eol_prompt(self):
        prompt = [{'role': 'user', 'content': EOL_PROMPTS['video']}]
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, use_image_id=False, max_slice_nums=2)
        return prompt


    def __init__(
            self, 
            model_name_or_path: str,
            load_llm: bool = False,
            device_map: Optional[Union[str, Dict[str, int]]] = None,
            **kwargs,
        ):

        MODEL_CLASS = self.LLM_CLASS if load_llm else self.MLLM_CLASS

        if load_llm:
            self.split_weights(model_name_or_path, model_name_or_path + '-llm')
            model_name_or_path += '-llm'

        attn_implementation = 'flash_attention_2' if device_map == 'cuda' else 'sdpa'

        self.is_llm = load_llm
        self.model = MODEL_CLASS.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    def split_weights(self, mllm_path, llm_path):
        if os.path.exists(llm_path):
            print(f'{llm_path} already exists. Skip splitting weights.')
            return
        print('Splitting LLM weights from MLLM.')
        model = self.MLLM_CLASS.from_pretrained(mllm_path)
        llm = model.llm
        processor = AutoProcessor.from_pretrained(mllm_path)
        tokenizer = AutoTokenizer.from_pretrained(mllm_path)
        llm.save_pretrained(llm_path)
        processor.save_pretrained(llm_path)
        tokenizer.save_pretrained(llm_path)


class BaseModelForInternVL2(BaseModel):
    
    ARCHITECTURE = "InternVLChatModel"
    LLM_CLASS = AutoModelForCausalLM
    MLLM_CLASS = AutoModel

    @property
    def describe_prompt(self):
        return "Describe the video in detail."

    @property
    def text_eol_prompt(self):
        prompt = f"<|im_start|>user\n{EOL_PROMPTS['text']}<|im_end|><|im_start|>assistant\n"
        return prompt
    
    @property
    def image_eol_prompt(self):
        prompt = f"<|im_start|>user\n{EOL_PROMPTS['image']}<|im_end|><|im_start|>assistant\n"
        return prompt
    
    @property
    def video_eol_prompt(self):
        prompt = f"<|im_start|>user\n{EOL_PROMPTS['video']}<|im_end|><|im_start|>assistant\n"
        return prompt

    def __init__(
            self, 
            model_name_or_path: str,
            load_llm: bool = False,
            device_map: Optional[Union[str, Dict[str, int]]] = None,
            **kwargs,
        ):

        MODEL_CLASS = self.LLM_CLASS if load_llm else self.MLLM_CLASS

        if load_llm:
            self.split_weights(model_name_or_path, model_name_or_path + '-llm')
            model_name_or_path += '-llm'

        self.is_llm = load_llm
        self.model = MODEL_CLASS.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )
             
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
        self.tokenizer.model_max_length = 16384
    
    def split_weights(self, mllm_path, llm_path):
        if os.path.exists(llm_path):
            print(f'{llm_path} already exists. Skip splitting weights.')
            return
        print('Splitting LLM weights from MLLM.')
        model = self.MLLM_CLASS.from_pretrained(mllm_path)
        llm = model.language_model
        tokenizer = AutoTokenizer.from_pretrained(mllm_path)
        llm.save_pretrained(llm_path)
        tokenizer.save_pretrained(llm_path)

    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

class BaseModelForLlavaNextVideo(BaseModel):

    ARCHITECTURE = "LlavaNextVideoForConditionalGeneration"
    LLM_CLASS = AutoModelForCausalLM
    MLLM_CLASS = LlavaNextVideoForConditionalGeneration

    @property
    def describe_prompt(self):
        return "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."

    @property
    def text_eol_prompt(self):
        prompt = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['text']}],
        }]
        prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt
    
    @property
    def image_eol_prompt(self):
        prompt = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['image']}],
        }]
        prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt
    
    @property
    def video_eol_prompt(self):
        prompt = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['video']}],
        }]
        prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt

    def __init__(
            self, 
            model_name_or_path: str,
            load_llm: bool = False,
            device_map: Optional[Union[str, Dict[str, int]]] = None,
            **kwargs,
        ):
        
        MODEL_CLASS = self.LLM_CLASS if load_llm else self.MLLM_CLASS

        if load_llm:
            self.split_weights(model_name_or_path, model_name_or_path + '-llm')
            model_name_or_path += '-llm'

        self.is_llm = load_llm  
        self.model = MODEL_CLASS.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def split_weights(self, mllm_path, llm_path):
        if os.path.exists(llm_path):
            print(f'{llm_path} already exists. Skip splitting weights.')
            return
        print('Splitting LLM weights from MLLM.')
        model = self.MLLM_CLASS.from_pretrained(mllm_path)
        llm = model.language_model
        processor = AutoProcessor.from_pretrained(mllm_path)
        tokenizer = AutoTokenizer.from_pretrained(mllm_path)
        llm.save_pretrained(llm_path)
        processor.save_pretrained(llm_path)
        tokenizer.save_pretrained(llm_path)

class BaseModelForTarsier(BaseModel):
    
    ARCHITECTURE = "TarsierForConditionalGeneration"
    LLM_CLASS = LlamaForCausalLM
    MLLM_CLASS = TarsierForConditionalGeneration

    @property
    def describe_prompt(self):
        return "Describe the video in detail."

    @property
    def text_eol_prompt(self):
        prompt = f'USER: {EOL_PROMPTS["text"]} ASSISTANT: '
        return prompt
    
    @property
    def image_eol_prompt(self):
        prompt = f'USER: {EOL_PROMPTS["image"]} ASSISTANT: '
        return prompt
    
    @property
    def video_eol_prompt(self):
        prompt = f'USER: {EOL_PROMPTS["video"]} ASSISTANT: '
        return prompt

    def __init__(
            self, 
            model_name_or_path: str,
            load_llm: Optional[bool] = None,
            device_map: Optional[Union[str, Dict[str, int]]] = None,
            **kwargs,
        ):

        MODEL_CLASS = self.LLM_CLASS if load_llm else self.MLLM_CLASS

        if load_llm:
            self.split_weights(model_name_or_path, model_name_or_path + '-llm')
            model_name_or_path += '-llm'
            model_config = None
            self.processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
        else:
            model_config = LlavaConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            self.processor = Processor(
                model_name_or_path,
                max_n_frames=32,
            )
        
        self.tokenizer = self.processor.tokenizer

        self.model = MODEL_CLASS.from_pretrained(
            model_name_or_path,
            config=model_config,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map=device_map,
            trust_remote_code=True
        )
        
        self.model.eval()

    def split_weights(self, mllm_path, llm_path):
        if os.path.exists(llm_path):
            print(f'{llm_path} already exists. Skip splitting weights.')
            return
        print('Splitting LLM weights from MLLM.')
        model = self.MLLM_CLASS.from_pretrained(mllm_path)
        llm = model.language_model
        processor = AutoProcessor.from_pretrained(mllm_path)
        tokenizer = AutoTokenizer.from_pretrained(mllm_path)
        llm.save_pretrained(llm_path)
        processor.save_pretrained(llm_path)
        tokenizer.save_pretrained(llm_path)

class BaseModelForQwen2VL(BaseModel):

    ARCHITECTURE = "Qwen2VLForConditionalGeneration"
    LLM_CLASS = Qwen2VLModel
    MLLM_CLASS = Qwen2VLForConditionalGeneration

    @property
    def describe_prompt(self):
        return "Describe the video in detail."

    @property
    def text_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['text']}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt
    
    @property
    def image_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['image']}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt
    
    @property
    def video_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['video']}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def __init__(
            self, 
            model_name_or_path: str,
            load_llm: Optional[bool] = None,
            device_map: Optional[Union[str, Dict[str, int]]] = None,
            **kwargs,
        ):        
        
        MODEL_CLASS = self.LLM_CLASS if load_llm else self.MLLM_CLASS

        self.load_llm = load_llm

        if load_llm:
            self.split_weights(model_name_or_path, model_name_or_path + '-llm')
            model_name_or_path += '-llm'
        
        self.model = MODEL_CLASS.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
             
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer

    def split_weights(self, mllm_path, llm_path):
        if os.path.exists(llm_path):
            print(f'{llm_path} already exists. Skip splitting weights.')
            return
        print('Splitting LLM weights from MLLM.')
        model = self.MLLM_CLASS.from_pretrained(mllm_path)
        llm = model.model
        processor = AutoProcessor.from_pretrained(mllm_path)
        tokenizer = AutoTokenizer.from_pretrained(mllm_path)
        llm.save_pretrained(llm_path)
        processor.save_pretrained(llm_path)
        tokenizer.save_pretrained(llm_path)
    
    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor


    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor


    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor
    
    def smart_resize(
        self, height: int, width: int, factor: int = qwen_vl_vision_process.IMAGE_FACTOR, min_pixels: int = qwen_vl_vision_process.MIN_PIXELS, max_pixels: int = qwen_vl_vision_process.MAX_PIXELS
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if max(height, width) / min(height, width) > qwen_vl_vision_process.MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {qwen_vl_vision_process.MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

#  The base model is the same as Qwen2-VL
class BaseModelForCaRe(BaseModelForQwen2VL):

    ARCHITECTURE = "CaReModel"