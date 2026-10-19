import base64
import json
import math
import os
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.error_sentinels import GENERATION_ERROR, VIDEO_DECODE_ERROR
from lmms_eval.models.model_utils.video_input import parse_video_input, validate_video_bounds
from lmms_eval.tasks._task_utils.eval_utils import extract_answer_letter

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Gemma 4 vision: soft-token budgets (see HF Gemma4 docs)
ALLOWED_MAX_SOFT_TOKENS = frozenset({70, 140, 280, 560, 1120})
DEFAULT_MAX_SOFT_TOKENS = 280
DEFAULT_VIDEO_MAX_SOFT_TOKENS = 70
DEFAULT_MAX_FRAMES = 128

def _extract_allowed_answers(context: str) -> set[str]:
    match = re.search(
        r"Respond with only the letter\s*\(([^)]*)\)\s*of the correct option",
        context,
        re.IGNORECASE,
    )
    if not match:
        return set()
    return {letter.upper() for letter in re.findall(r"\b([A-N])\b", match.group(1), re.IGNORECASE)}


def _extract_unambiguous_answer(text: str, context: str) -> Optional[str]:
    allowed_answers = _extract_allowed_answers(context)
    if not allowed_answers:
        return None

    answer = extract_answer_letter(text)
    return answer if answer in allowed_answers else None


@register_model("gemma4")
class Gemma4(lmms):
    """
    Gemma 4 multimodal model (Hugging Face `AutoModelForImageTextToText`).

    Default checkpoint is the latest flagship instruction-tuned release; override with
    ``pretrained`` (e.g. ``google/gemma-4-26B-A4B-it``, ``google/gemma-4-E2B-it``).

    Requires a recent ``transformers`` build with Gemma 4 support.
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-4-31B-it",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
        video_max_soft_tokens: int = DEFAULT_VIDEO_MAX_SOFT_TOKENS,
        max_frames: int = DEFAULT_MAX_FRAMES,
        interleave_visuals: Optional[bool] = False,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        enable_thinking: bool = False,
        reasoning_prompt: Optional[str] = None,
        prediction_output_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if max_soft_tokens not in ALLOWED_MAX_SOFT_TOKENS:
            raise ValueError(f"max_soft_tokens must be one of {sorted(ALLOWED_MAX_SOFT_TOKENS)}, got {max_soft_tokens}")
        if video_max_soft_tokens not in ALLOWED_MAX_SOFT_TOKENS:
            raise ValueError(
                f"video_max_soft_tokens must be one of {sorted(ALLOWED_MAX_SOFT_TOKENS)}, got {video_max_soft_tokens}"
            )
        max_frames = int(max_frames)
        if max_frames < 1:
            raise ValueError(f"max_frames must be positive, got {max_frames}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        self._tokenizer = self.processor.tokenizer

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.enable_thinking = enable_thinking
        self.prediction_output_dir = Path(prediction_output_dir).expanduser() if prediction_output_dir else None
        if self.prediction_output_dir is not None:
            self.prediction_output_dir.mkdir(parents=True, exist_ok=True)
            eval_logger.info(f"Saving per-sample predictions to {self.prediction_output_dir}")

        self.max_soft_tokens = max_soft_tokens
        self.video_max_soft_tokens = video_max_soft_tokens
        self.max_frames = max_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

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
        self.model.eval()

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

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
        raise NotImplementedError("Not implemented for Gemma4.")

    def flatten(self, input: List[List]) -> List:
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _encode_image_data_url(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _sanitize_prediction_component(value) -> str:
        return str(value).replace("/", "_").replace("\\", "_")

    def _prediction_path(self, task: str, split: str, doc_id: int) -> Optional[Path]:
        if self.prediction_output_dir is None:
            return None
        doc = self.task_dict[task][split][doc_id]
        db = self._sanitize_prediction_component(doc.get("db", task))
        qid = self._sanitize_prediction_component(doc.get("qid", doc_id))
        return self.prediction_output_dir / f"{db}**@@**{qid}.json"

    def _load_chunk_predictions(self, task: str, split: str, doc_ids) -> Optional[List[str]]:
        if self.prediction_output_dir is None:
            return None

        predictions = []
        for doc_id in doc_ids:
            path = self._prediction_path(task, split, doc_id)
            try:
                with path.open("r", encoding="utf-8") as file:
                    saved = json.load(file)
                prediction = saved.get("pred")
                raw_prediction = saved.get("raw_pred")
                if (
                    not isinstance(prediction, str)
                    or prediction.startswith("error::")
                    or not isinstance(raw_prediction, str)
                ):
                    return None
                predictions.append(prediction)
            except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
                return None
        return predictions

    def _save_prediction(
        self,
        task: str,
        split: str,
        doc_id: int,
        prediction: str,
        raw_prediction: Optional[str] = None,
    ) -> None:
        path = self._prediction_path(task, split, doc_id)
        if path is None:
            return

        output = dict(self.task_dict[task][split][doc_id])
        output["pred"] = prediction
        output["raw_pred"] = raw_prediction if raw_prediction is not None else prediction
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(output, file, ensure_ascii=False, indent=4, default=str)
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            eval_logger.exception(f"Failed to save prediction: {path}")

    def _decode_video(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> tuple[np.ndarray, VideoMetadata]:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        video_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        validate_video_bounds(start_time, end_time, total_frames, video_fps)

        start_frame = 0 if start_time is None else int(math.floor(start_time * video_fps))
        end_frame = total_frames - 1 if end_time is None else int(math.ceil(end_time * video_fps)) - 1
        start_frame = min(total_frames - 1, max(0, start_frame))
        end_frame = min(total_frames - 1, max(0, end_frame))
        if start_frame > end_frame:
            raise ValueError(f"Video segment contains no frames: {video_path} [{start_time}, {end_time}]")

        segment_duration = (end_frame - start_frame + 1) / video_fps
        num_frames = max(1, min(self.max_frames, int(segment_duration)))
        frame_indices = np.linspace(start_frame, end_frame, num_frames).round().astype(np.int64)
        frames = vr.get_batch(frame_indices).asnumpy()
        segment_total_frames = end_frame - start_frame + 1
        relative_frame_indices = frame_indices - start_frame
        metadata = VideoMetadata(
            total_num_frames=segment_total_frames,
            fps=video_fps,
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            duration=segment_total_frames / video_fps,
            video_backend="decord",
            frames_indices=relative_frame_indices.tolist(),
        )
        return frames, metadata

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = all_gen_kwargs[0]

            cached_answers = self._load_chunk_predictions(task, split, doc_id)
            if cached_answers is not None:
                for ans, context in zip(cached_answers, contexts):
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)
                continue

            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            failure_sentinel = VIDEO_DECODE_ERROR
            try:
                batched_messages = []
                video_metadatas = []
                for i, context in enumerate(contexts):
                    if "<image>" in context:
                        context = context.replace("<image>", "")

                    message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

                    if self.reasoning_prompt:
                        context = context.strip() + self.reasoning_prompt
                        contexts[i] = context

                    processed_visuals = []
                    for visual in visual_list[i]:
                        video_spec = parse_video_input(visual)
                        if video_spec is not None and video_spec.path.lower().endswith((".mp4", ".avi", ".mov", ".webm")):
                            if not os.path.exists(video_spec.path):
                                raise FileNotFoundError(f"Video file not found: {video_spec.path}")
                            video_input, video_metadata = self._decode_video(
                                video_spec.path,
                                video_spec.start_time,
                                video_spec.end_time,
                            )
                            processed_visuals.append({"type": "video", "video": video_input})
                            video_metadatas.append(video_metadata)
                        elif isinstance(visual, Image.Image):
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "url": self._encode_image_data_url(visual),
                                }
                            )
                        else:
                            raise TypeError(f"Unsupported visual input: {type(visual).__name__}")

                    if not processed_visuals:
                        raise ValueError("No valid visual input was provided")

                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                    batched_messages.append(message)

                # HF Processor.apply_chat_template: only Jinja template variables may appear in **kwargs.
                # Tokenizer / vision options must go in processor_kwargs (see processing_utils.apply_chat_template).
                inputs = self.processor.apply_chat_template(
                    batched_messages,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={
                        "padding": True,
                        "images_kwargs": {"max_soft_tokens": self.max_soft_tokens},
                        "videos_kwargs": {
                            "do_sample_frames": False,
                            "max_soft_tokens": self.video_max_soft_tokens,
                            "video_metadata": video_metadatas,
                        },
                    },
                )

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                default_gen_kwargs = {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

                if current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                failure_sentinel = GENERATION_ERROR
                cont = self.model.generate(
                    **inputs,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                raw_answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                answers = []
                raw_predictions = []
                for raw_answer, input_ids, context in zip(raw_answers, inputs.input_ids, contexts):
                    parsed = self.processor.parse_response(raw_answer, prefix=input_ids)
                    content = (parsed.get("content") or "").strip()
                    raw_predictions.append(raw_answer.strip())
                    final_answer = _extract_unambiguous_answer(content, context)
                    answers.append(final_answer or content)
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans
            except Exception as e:
                eval_logger.exception(f"{failure_sentinel}: {e}")
                answers = [failure_sentinel] * len(contexts)
                raw_predictions = list(answers)

            for ans, raw_prediction, context, current_doc_id in zip(answers, raw_predictions, contexts, doc_id):
                self._save_prediction(task, split, current_doc_id, ans, raw_prediction)
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
