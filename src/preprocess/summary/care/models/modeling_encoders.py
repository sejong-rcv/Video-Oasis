import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import (
    ToPILImage,
    functional,
)
from typing import Dict, List, Optional, Union
import os

from abc import ABCMeta, abstractmethod

from care.models.modeling_basemodels import (
    BaseModelForMiniCPMV,
    BaseModelForLlavaNextVideo,
    BaseModelForTarsier,
    BaseModelForQwen2VL,
    BaseModelForInternVL2,
    BaseModelForCaRe,
)
from care.utils.model import load_architectures_from_config, transform_pixel_values

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

encoder_registry = {}

class EncodeMixin(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            encoder_registry[cls.ARCHITECTURE] = cls

    @abstractmethod
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        """
        Encodes vision data (images or videos) into a tensor representation.

        Args:
            pixel_values (torch.Tensor | List[torch.Tensor]): The input pixel values. 
                - If a tensor, it should be of shape (B, C, H, W) for images or (B, T, C, H, W) for videos.
                - If a list, it will be stacked into a tensor.

        Returns:
            torch.Tensor: The encoded tensor representation of the input vision data.

        Raises:
            ValueError: If `pixel_values` is not 4D or 5D.

        ## Notes:
            - This function does not accept unbatched inputs.
            - `pixel_values` should be of type uint8.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        """
        Encodes the given text(s) into a tensor representation using the model.

        Args:
            text (str | List[str]): A single string or a list of strings to be encoded.

        Returns:
            torch.Tensor: The tensor representation of the encoded text(s).

        ## Notes:
            - The method uses a prompt to encode the text.
            - If a single string is provided, it is converted into a list containing that string.
            - The method processes the prompts and generates the tensor representation using the model.
            - The output tensor contains the hidden states of the last token for each input text.
        """
        raise NotImplementedError

class AutoEncoder:
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
        if model_arch not in encoder_registry:
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
        
        MODEL_CLASS = encoder_registry[model_arch]

        return MODEL_CLASS.from_pretrained(model_name_or_path, load_llm=False, device_map=device_map, **kwargs)
    

class EncoderForMiniCPMV(BaseModelForMiniCPMV, EncodeMixin):
    
    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        
        prompt = self.text_eol_prompt
        # print(prompt)
        if isinstance(text, str):
            text = [text]
        prompts = [prompt.replace('<sent>', t) for t in text]
        inputs = self.processor(prompts, [[]]*len(prompts), return_tensors="pt").to('cuda')
        inputs.pop("image_sizes")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, tokenizer=self.tokenizer, max_new_tokens=1, decode_text=False, repetition_penalty=1.2, output_hidden_states=True, return_dict_in_generate=True)
        return outputs.hidden_states[0][-1][:, -1, :]
    
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:

        pixel_values = transform_pixel_values(pixel_values)
        T = pixel_values.shape[1]

        if T == 1:
            # if the input is image
            prompt = self.image_eol_prompt
            # replace <image>\n to (<image>./</image>)\n
            prompts = [prompt.replace(f"<image>\n", f"(<image>./</image>)\n") for p in pixel_values]
        else:
            # if the input is video
            prompt = self.video_eol_prompt
            # replace <video>\n to N * (<image>./</image>)\n
            prompts = [prompt.replace(f"<video>\n", f"(<image>./</image>)\n" * len(p)) for p in pixel_values]

        inputs = self.processor(prompts, pixel_values, return_tensors="pt").to('cuda')
        inputs.pop("image_sizes")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, tokenizer=self.tokenizer, max_new_tokens=1, decode_text=False, repetition_penalty=1.2, output_hidden_states=True, return_dict_in_generate=True)

        return outputs.hidden_states[0][-1][:, -1, :]


class EncoderForInternVL2(BaseModelForInternVL2, EncodeMixin):

    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        self.model.eval()
        prompt = self.text_eol_prompt
        if isinstance(text, str):
            text = [text]
        prompts = [prompt.replace('<sent>', t) for t in text]

        # to avoid img_context_token_id assertion error
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.tokenizer.padding_side = 'left'
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        outputs = self.model.generate(
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=True,
            eos_token_id=eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        return outputs.hidden_states[0][-1][:, -1, :]

    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        pixel_values = transform_pixel_values(pixel_values)
        T = pixel_values.shape[1]

        if T == 1:
            # if the input is image
            prompt = self.image_eol_prompt
            dynamic_preprocess_max_num = 12
        else:
            # if the input is video
            prompt = self.video_eol_prompt
            dynamic_preprocess_max_num = 1
        
        self.model.eval()
        IMG_START_TOKEN='<img>'
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id
        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        transform = self.build_transform(input_size=448)
        output_embs = []
        
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

            # num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
                prompt = prompt.replace('<image>', image_tokens, 1)

            model_inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = model_inputs['input_ids'].to(self.model.device)
            attention_mask = model_inputs['attention_mask'].to(self.model.device)
            outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=True,
                eos_token_id=eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_embs.append(outputs.hidden_states[0][-1][:, -1, :])

        return torch.cat(output_embs)
    
class EncoderForLlavaNextVideo(BaseModelForLlavaNextVideo, EncodeMixin):
    
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:

        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values)
        
        images = None
        videos = None

        if pixel_values.ndim == 4:
            # pixel_values is (B, C, H, W)
            # (B, C, H, W) -> (B, 1, C, H, W)
            prompt = self.image_eol_prompt
            images = list(pixel_values)
            # print(prompt)
        elif pixel_values.ndim == 5:
            # pixel_values is (B, T, C, H, W)
            prompt = self.video_eol_prompt
            videos = list(pixel_values)
            # print(prompt)
        else:
            raise ValueError(f"pixel_values should be 4D or 5D, got {pixel_values.ndim}D")
        inputs = self.processor(prompt, images=images, videos=videos, padding=True, return_tensors="pt").to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)

        return outputs.hidden_states[0][-1][:, -1, :]

    def encode_text(self, text: str | List[str]) -> torch.Tensor:

        prompt = self.text_eol_prompt

        if isinstance(text, str):
            text = [text]
        
        prompts = [prompt.replace('<sent>', t) for t in text]
        inputs = self.processor(prompts, padding=True, return_tensors="pt").to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        return outputs.hidden_states[0][-1][:, -1, :]

class EncoderForTarsier(BaseModelForTarsier, EncodeMixin):

    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:

        pixel_values = transform_pixel_values(pixel_values) # [B, T, C, H, W]
        nframes = pixel_values.shape[1]
        prompt = self.image_eol_prompt if nframes == 1 else self.video_eol_prompt
        
        to_image = ToPILImage()
        batched_frames = []
        for batch in pixel_values:
            frames = [to_image(v) for v in batch]
            batched_frames.append(frames)

        generate_kwargs = {
            "max_new_tokens": 1,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }

        vision_embs = []

        for frames in batched_frames:
            input_prompt = prompt.replace("<video>", "<image>"*len(frames))
            input_ids = self.processor.get_text_inputs(input_prompt)
            frames = self.processor.get_pixel_values(frames)
            inputs = {
                "input_ids": input_ids,
                "pixel_values": frames
            }
            inputs = {k:v.to(self.model.device) for k,v in inputs.items() if v is not None}
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            vision_embs.append(outputs.hidden_states[0][-1][:, -1, :])
        
        vision_embs = torch.cat(vision_embs)
        return vision_embs
    
    def encode_text(self, text: str | List[str]) -> torch.Tensor:

        prompt = self.text_eol_prompt

        if isinstance(text, str):
            text = [text]
        
        prompts = [prompt.replace('<sent>', t) for t in text]

        generate_kwargs = {
            "max_new_tokens": 1,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }

        text_embs = []

        for p in prompts:
            text_inputs = self.processor.get_text_inputs(p)
            inputs = {
                "input_ids": text_inputs,
            }
            inputs = {k:v.to(self.model.device) for k,v in inputs.items() if v is not None}
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            text_embs.append(outputs.hidden_states[0][-1][:, -1, :])
        
        text_embs = torch.cat(text_embs)
        return text_embs
    
class EncoderForQwen2VL(BaseModelForQwen2VL, EncodeMixin):
    
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        
        batched_pixel_values = transform_pixel_values(pixel_values)
        vision_embs = []
        prompt = self.video_eol_prompt
        prompt = prompt.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")

        for pixel_values in batched_pixel_values:
        
            nframes, _, height, width = pixel_values.shape
            min_pixels = VIDEO_MIN_PIXELS
            total_pixels = VIDEO_TOTAL_PIXELS
            max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = 230400
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pixel_values = functional.resize(
                pixel_values,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            
            inputs = self.processor(
                text=[prompt],
                images=None,
                videos=[pixel_values],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
            vision_embs.append(output.hidden_states[0][-1][:, -1, :])
        vision_embs = torch.cat(vision_embs)
        return vision_embs
    
    def encode_text(self, text: str | List[str]) -> torch.Tensor:

        prompt = self.text_eol_prompt

        if isinstance(text, str):
            text = [text]
        prompts = [prompt.replace('<sent>', t) for t in text]
            
        inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        return output.hidden_states[0][-1][:, -1, :]

class EncoderForCaRe(BaseModelForCaRe, EncodeMixin):
    
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        
        batched_pixel_values = transform_pixel_values(pixel_values)
        batched_pixel_values = torch.repeat_interleave(batched_pixel_values, repeats=2, dim=1)
        vision_embs = []
        prompt = self.video_eol_prompt
        prompt = prompt.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")

        for pixel_values in batched_pixel_values:
        
            nframes, _, height, width = pixel_values.shape
            min_pixels = VIDEO_MIN_PIXELS
            total_pixels = VIDEO_TOTAL_PIXELS
            max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = 230400
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pixel_values = functional.resize(
                pixel_values,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            
            inputs = self.processor(
                text=[prompt],
                images=None,
                videos=[pixel_values],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
            vision_embs.append(output.hidden_states[0][-1][:, -1, :])
        vision_embs = torch.cat(vision_embs)
        return vision_embs
    
    def encode_text(self, text: str | List[str]) -> torch.Tensor:

        prompt = self.text_eol_prompt

        if isinstance(text, str):
            text = [text]
        prompts = [prompt.replace('<sent>', t) for t in text]
            
        inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        return output.hidden_states[0][-1][:, -1, :]
