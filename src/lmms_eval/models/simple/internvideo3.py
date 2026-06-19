import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import logging
from datetime import timedelta
from typing import List, Tuple

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.vision_process import process_vision_info

torch.set_num_threads(1)

eval_logger = logging.getLogger("eval_logger")

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

VALID_GENERATE_KEYS = {
    "max_new_tokens",
    "min_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "num_beams",
    "repetition_penalty",
    "length_penalty",
    "use_cache",
    "eos_token_id",
    "pad_token_id",
}


@register_model("internvideo3")
class InternVideo3(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVideo2_5_Chat_8B",
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        max_frames_num: int = 32,
        fps: float = 1.0,
        video_min_pixels: int = 256 * 28 * 28,
        video_max_pixels: int = 2048 * 28 * 28,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.fps = fps
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if device_map == "auto" and accelerator.num_processes == 1:
            load_device_map = "auto"
        else:
            load_device_map = {"": str(self._device)}

        self._model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=load_device_map,
            trust_remote_code=True,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            self.path,
            trust_remote_code=True,
        )

        self._config = self._model.config
        self._tokenizer = getattr(self._processor, "tokenizer", None)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP, FSDP, and DeepSpeed are supported."

            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                deepspeed_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True,
                    **deepspeed_kwargs,
                )
                eval_logger.info(
                    "Detected DistributedType.DEEPSPEED. "
                    "Make sure accelerate config uses ZeRO stage 0 for evaluation."
                )

            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(
                    self._model,
                    evaluation_mode=True,
                )

            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes

            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism.")

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info("Using single process with device_map='auto'.")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}.")
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

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

    def flatten(self, input_list):
        return [item for sublist in input_list for item in sublist]

    def _prepare_generate_kwargs(self, gen_kwargs):
        gen_kwargs = dict(gen_kwargs)

        until = gen_kwargs.pop("until", None)
        if isinstance(until, str):
            until = [until]
        elif until is not None:
            until = list(until)

        if "max_gen_toks" in gen_kwargs and "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_gen_toks")
        else:
            gen_kwargs.pop("max_gen_toks", None)

        for key, value in DEFAULT_GEN_KWARGS.items():
            gen_kwargs.setdefault(key, value)

        generate_kwargs = {
            key: value
            for key, value in gen_kwargs.items()
            if key in VALID_GENERATE_KEYS and value is not None
        }

        generate_kwargs.setdefault("max_new_tokens", 1024)
        generate_kwargs.setdefault("use_cache", True)

        if generate_kwargs.get("temperature", None) == 0:
            generate_kwargs["do_sample"] = False
            generate_kwargs.pop("temperature", None)

        if generate_kwargs.get("do_sample") is False:
            generate_kwargs.pop("temperature", None)

        return generate_kwargs, until

    def _apply_until(self, response, until):
        if until is None:
            return response.strip()

        for term in until:
            if term:
                response = response.split(term)[0]

        return response.strip()

    def _build_messages(self, contexts, visuals):
        if self.modality == "image":
            content = []

            for visual in visuals:
                content.append(
                    {
                        "type": "image",
                        "image": visual,
                    }
                )

            content.append(
                {
                    "type": "text",
                    "text": contexts,
                }
            )

            return [
                {
                    "role": "user",
                    "content": content,
                }
            ]

        if self.modality == "video":
            video_path = visuals[0]

            if len(visuals) > 1 and isinstance(visuals[1], dict):
                media_dict = visuals[1]
            else:
                media_dict = {}

            fps = media_dict.get("fps", self.fps)
            if fps is not None:
                fps = min(float(fps), float(self.fps))

            video_content = {
                "type": "video",
                "video": video_path,
                "fps": fps,
                "min_pixels": self.video_min_pixels,
                "max_pixels": self.video_max_pixels,
            }

            return [
                {
                    "role": "user",
                    "content": [
                        video_content,
                        {
                            "type": "text",
                            "text": contexts,
                        },
                    ],
                }
            ]

        raise ValueError(f"Unsupported modality: {self.modality}")

    def _build_inputs(self, messages):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
            )

            processor_kwargs = dict(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            processor_kwargs.update(video_kwargs)

        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)

            processor_kwargs = dict(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        inputs = self.processor(**processor_kwargs)
        return inputs.to(self._device)

    def _decode_output(self, inputs, output):
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output)
        ]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            try:
                generate_kwargs, until = self._prepare_generate_kwargs(gen_kwargs)

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)

                messages = self._build_messages(contexts, visuals)
                inputs = self._build_inputs(messages)

                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        **generate_kwargs,
                    )

                response = self._decode_output(inputs, output)
                response = self._apply_until(response, until)
            except Exception as e:
                eval_logger.exception(
                    f"Error during generation. {e}")
                response = "NONE"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()   

            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")