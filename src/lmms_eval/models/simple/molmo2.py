import base64
import re
import os
import json
from io import BytesIO
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.vision_process import process_vision_info


@register_model("molmo2")
class Molmo2(lmms):

    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-8B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = True,
        video_min_pixels: Optional[int] = 16 * 32 * 32,
        video_max_pixels: Optional[int] = 256 * 32 * 32,
        video_total_pixels: Optional[int] = 128000 * 32 * 32,
        min_frames: Optional[int] = 4,
        max_frames: Optional[int] = 128,
        nframes: Optional[int] = 128,
        fps: Optional[float] = 2.0,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        sampling: Optional[str] = "uniform",
        **kwargs,
    ) -> None:
        """
        Args:
            video_min_pixels (int, optional): The minimum pixels number for a frame of a video. Defaults to 128 * 28 * 28.
            video_max_pixels (int, optional): The maximum pixels number for a frame of a video. Defaults to 768 * 28 * 28.
            video_total_pixels (int, optional): The total pixels number for a video. Defaults to 115200 * 28 * 28.
            min_frames (int, optional): The minimum frames number allowed for a video. Defaults to 4.
            max_frames (int, optional): The maximum frames number allowed for a video. Defaults to 768.
            nframes (int, optional): The exact number of frames to extract for a video. Defaults to None, meaning the frame number is decided by fps and max_frames.
            fps (float, optional): The fps to extract frames for a video. Defaults to None.
        """
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        # Add attention implementation if specified
        if use_flash_attention_2 is not None:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()

        # data configuration for fetch_video
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_total_pixels = video_total_pixels
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.nframes = nframes
        self.fps = fps
        self.sampling = sampling
        eval_logger.info(
            f"video_min_pixels: {self.video_min_pixels}, "
            f"video_max_pixels: {self.video_max_pixels}, "
            f"video_total_pixels: {self.video_total_pixels}, "
            f"min_frames: {self.min_frames}, "
            f"max_frames: {self.max_frames}, "
            f"nframes: {self.nframes}, "
            f"fps: {self.fps}"
        )

        self.system_prompt = system_prompt
        eval_logger.info(f"system_prompt: {self.system_prompt}")

        self.processor = AutoProcessor.from_pretrained(pretrained, dtype='bfloat16',device_map=self.device_map,trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        
        self.interleave_visuals = interleave_visuals

        self.mode = pretrained.split('-')[-1]

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)

            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}"
                )

            # Avoid using '\n\n' as a stopper for Qwen3_VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            try:
                batched_messages = []
                for i, context in enumerate(contexts):
                    if "<image>" in context:
                        context = context.replace("<image>", "")

                    message = []

                    processed_visuals = []
                    if visual_list[i] is not None:
                        for visual in visual_list[i]:
                            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".webm", ".MP4")):  # Video file
                                visual_dict = {
                                    "type": "video",
                                    "video": visual,
                                    "min_pixels": self.video_min_pixels,
                                    "max_pixels": self.video_max_pixels,
                                    "total_pixels": self.video_total_pixels,
                                    "min_frames": self.min_frames,
                                    "max_frames": self.max_frames,
                                    "fps": self.fps,
                                }
                                if self.nframes is not None:
                                    visual_dict["nframes"] = self.nframes
                                    visual_dict.pop("fps")

                                processed_visuals.append(visual_dict)

                    if self.interleave_visuals is False:
                        message.append(
                            {
                                "role": "user",
                                "content": processed_visuals + [{"type": "text", "text": context}],
                            }
                        )
                    batched_messages.append(message)

                texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    batched_messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                    sampling=self.sampling, top_idx=None)

                if video_inputs is not None:
                    video_inputs, video_metadatas = zip(*video_inputs)
                    video_inputs, video_metadatas = (
                        list(video_inputs),
                        list(video_metadatas),
                    )
                else:
                    video_metadatas = None

                padding_side = "left" if self.batch_size > 1 else "right"

                video_inputs[0] = video_inputs[0].permute(0, 2, 3, 1)
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    do_resize=False,
                    padding=True,
                    padding_side=padding_side,
                    return_tensors="pt",
                    **video_kwargs,
                )
                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                # Set default generation kwargs
                default_gen_kwargs = {
                    "max_new_tokens": 32768,
                    "temperature": 0.0,  # Set to 0 for greedy default
                    "top_p": None,
                    "num_beams": 1,
                }
                # Update with provided kwargs
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
                pad_token_id = self.tokenizer.pad_token_id

                if current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans.split('</think>')[-1].strip()

                for ans, context in zip(answers, contexts):
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

            except Exception as e:
                print(e)
                answers = ['NONE']
                for ans, context in zip(answers, contexts):
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
