import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import time
import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig
import os
import glob
from PIL import Image

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.error_sentinels import GENERATION_ERROR, VIDEO_DECODE_ERROR
from lmms_eval.models.model_utils.load_video import read_video_pyav
from lmms_eval.models.model_utils.video_input import parse_video_input, validate_video_bounds

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
# eval_logger = logging.getLogger("lmms-eval")
from loguru import logger as eval_logger

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

@register_model("llava_llada_video")
class LlavaLLaDAVideo(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        attn_implementation: Optional[str] = best_fit_attn_implementation, # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "llava_llada",
        use_cache: Optional[bool] = False,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        mm_newline_position: str = "grid",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        video_fps: int = 1,
        force_sample: bool = False,
        add_time_instruction: bool = False,
        ## add for LLaDA, Not used for now, this is for likelihood evaluation, not used for generation
        mc_num: Optional[int] = 128,
        batch_size_ppl: Optional[int] = 32,
        is_check_greedy: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        
        # -- LLaDA
        self.mc_num = mc_num
        self.batch_size_ppl = batch_size_ppl
        self.is_check_greedy = is_check_greedy
        assert mc_num % self.batch_size_ppl == 0
        print(f'model: {pretrained}')
        print(f'Is check greedy: {is_check_greedy}')
        
        self.video_fps = video_fps
        self.force_sample = force_sample
        self.add_time_instruction = add_time_instruction
        print("force sample:", self.force_sample)
        
        self.torch_dtype = torch_dtype
        if isinstance(self.torch_dtype, str):
            if self.torch_dtype == "bfloat16":
                self.torch_dtype = torch.bfloat16
            elif self.torch_dtype == "float16":
                self.torch_dtype = torch.float16
            elif self.torch_dtype == "float32":
                self.torch_dtype = torch.float32

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.mm_newline_position = mm_newline_position
        
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        overwrite_config["mm_newline_position"] = self.mm_newline_position

        llava_model_args["overwrite_config"] = overwrite_config
        llava_model_args["torch_dtype"] = torch_dtype
        
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)

        self._config = self._model.config
        self.model.eval()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
            
        if self._rank == 0:
            print(self._model)

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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])
    
    @torch.no_grad()
    def suffix_greedy_prediction(self, input_ids, prompt_input_ids, images, **kwargs):
        if not self.is_check_greedy:
            return False
        # **kwargs: contains image_sizes
        generated_length = input_ids.shape[1] - prompt_input_ids.shape[1]
        cont = self.model.generate(prompt_input_ids, images=images, 
            steps=generated_length, gen_length=generated_length, block_length=generated_length, 
            **kwargs)
        
        cont_toks = input_ids[:, prompt_input_ids.shape[1] :] 
        correct = (cont == cont_toks).all()

        return correct

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError()

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num, fps, force_sample=False, bound=None):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_fps = float(vr.get_avg_fps())
        start_idx, end_idx = 0, total_frame_num - 1
        if bound is not None:
            validate_video_bounds(bound[0], bound[1], total_frame_num, video_fps)
            start_idx = max(0, min(math.ceil(bound[0] * video_fps), total_frame_num - 1))
            end_idx = max(start_idx, min(math.floor(bound[1] * video_fps), total_frame_num - 1))

        video_time = (end_idx - start_idx + 1) / video_fps
        frame_step = max(1, round(video_fps / fps))
        frame_idx = list(range(start_idx, end_idx + 1, frame_step))
        frame_time = [(i - start_idx) / video_fps for i in frame_idx]
        target_frame_count = max(1, min(int(video_time * fps), max_frames_num))
        if len(frame_idx) != target_frame_count or force_sample:
            uniform_sampled_frames = np.linspace(start_idx, end_idx, target_frame_count, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [(i - start_idx) / video_fps for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()

        return spare_frames, frame_time, video_time

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)
        num_tokens = 0

        start_time = time.time()
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "cfg" not in gen_kwargs:
                gen_kwargs["cfg"] = 0.
            if "remasking" not in gen_kwargs:
                gen_kwargs["remasking"] = "low_confidence"
            if "gen_length" not in gen_kwargs:
                gen_kwargs["gen_length"] = 2
            if "block_length" not in gen_kwargs:
                gen_kwargs["block_length"] = 2
            if "gen_steps" not in gen_kwargs and "steps" not in gen_kwargs:
                gen_kwargs["steps"] = 2
            elif "gen_steps" in gen_kwargs and "steps" not in gen_kwargs:
                gen_kwargs["steps"] = gen_kwargs["gen_steps"]

            think_mode = gen_kwargs.pop("think_mode", None)

            question_input = []
            decode_failed = False
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=self.torch_dtype, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=self.torch_dtype, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=self.torch_dtype, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=self.torch_dtype, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif isinstance(visual[0], (str, tuple)):  # For video task
                        image_tensor = []
                        video_spec = parse_video_input(visual[0]) if len(visual) == 1 else None
                        video_bound = None
                        if video_spec is not None:
                            visual = [video_spec.path]
                            if video_spec.has_time_bound:
                                video_bound = (video_spec.start_time, video_spec.end_time)

                        if os.path.isdir(visual[0]):
                            visuals = glob.glob(visual[0] + "/*")
                        else:
                            visuals = [visual[0]]
                        try:
                            if len(visuals) == 1:
                                if self.video_decode_backend == "decord":
                                    frames, frame_time, video_time = self.load_video(
                                        visual[0],
                                        self.max_frames_num,
                                        self.video_fps,
                                        force_sample=self.force_sample,
                                        bound=video_bound,
                                    )
                                else:
                                    raise NotImplementedError()
                            else:
                                if "mvbench" in task:
                                    fps = 3
                                    video_time = len(visuals) / fps
                                    sampled_indices = np.linspace(0, len(visuals) - 1, self.max_frames_num, dtype=int)
                                    frame_idx = sampled_indices.tolist()
                                    frame_time = [i / fps for i in frame_idx]
                                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
                                    frames = np.stack([np.array(Image.open(visuals[i])) for i in frame_idx], axis=0)
                                else:
                                    raise ValueError(f"Input type not recognized {visual[0]}")
                        
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(
                                device=self.device,
                                dtype=self.torch_dtype,
                                non_blocking=True,
                            )
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.exception(f"{VIDEO_DECODE_ERROR}: {e}")
                            decode_failed = True
                            break

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three scenarios:
                    1. No image, and therefore, no image token should be added.
                    2. Image token is already specified in the context, so we don't need to add it.
                    3. Image token is not specified in the context and there are image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    
                    if self.add_time_instruction:
                        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
                        context = f"{time_instruciton}\n{context}"
                    question = image_tokens + "\n" + context
                    
                else:
                    question = context

                if think_mode == "no_think":
                    question = question + "/no_think"
                elif think_mode == "think":
                    question = question + "/think"
                elif think_mode is not None:
                    raise NotImplementedError(f"Unsupported think_mode: {think_mode}")
                    
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" or "llava_llada" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            if decode_failed:
                res.append(VIDEO_DECODE_ERROR)
                self.cache_hook.add_partial(
                    "generate_until",
                    (batched_contexts[0], gen_kwargs),
                    VIDEO_DECODE_ERROR,
                )
                pbar.update(1)
                continue
                
            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                if "stopping_criteria" in gen_kwargs:
                    if isinstance(gen_kwargs["stopping_criteria"], str):
                        gen_kwargs["stopping_criteria"] = [gen_kwargs["stopping_criteria"]]
                    if stop_str not in gen_kwargs["stopping_criteria"]:
                        gen_kwargs["stopping_criteria"].extend(keywords)
                else:
                    gen_kwargs["stopping_criteria"] = keywords

                gen_kwargs["tokenizer"] = self.tokenizer
                
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                gen_kwargs["modalities"] = ["video"]
                if "stopping_criteria" in gen_kwargs:
                    if isinstance(gen_kwargs["stopping_criteria"], str):
                        gen_kwargs["stopping_criteria"] = [gen_kwargs["stopping_criteria"]]
                    if stop_str not in gen_kwargs["stopping_criteria"]:
                        gen_kwargs["stopping_criteria"].extend(keywords)
                else:
                    gen_kwargs["stopping_criteria"] = keywords
                
                gen_kwargs["tokenizer"] = self.tokenizer
                
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=False, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                text_outputs = [output[:-1] if output.endswith('.') else output for output in text_outputs]
                num_tokens += (cont != self.tokenizer.eos_token_id).sum()

            except Exception as e:
                eval_logger.exception(f"{GENERATION_ERROR}: {e}")
                res.append(GENERATION_ERROR)
                self.cache_hook.add_partial(
                    "generate_until",
                    (batched_contexts[0], gen_kwargs),
                    GENERATION_ERROR,
                )
                pbar.update(1)
                continue

            text_outputs = [response.strip() for response in text_outputs]
            
            res.extend(text_outputs)
            eval_logger.debug(f"Question: {question_input}")
            eval_logger.debug(f"Answer: {text_outputs}")
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
        print(f"Total number of tokens: {num_tokens}")
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for `LlavaLLaDAVideo`")
