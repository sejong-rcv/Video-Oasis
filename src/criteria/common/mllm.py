"""Shared loading and generation helpers for multimodal models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image


MediaType = Literal["image", "video"]


@dataclass
class MLLMRuntime:
    name: str
    model: Any
    processor: Any
    device: str
    trim_input_tokens: bool


def load_mllm(name: str, device: str = "cuda") -> MLLMRuntime:
    import torch
    from transformers import (
        AutoModel,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    from .models import get_model_entry

    entry = get_model_entry(name)
    if entry.backend != "mllm":
        raise ValueError(f"Model {name!r} is not an MLLM (backend={entry.backend!r}).")

    if name == "qwen25_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            entry.path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(entry.path)
        trim_input_tokens = True
    elif name in ("qwen3_vl", "videoauto_r1_qwen3"):
        model_class = Qwen3VLForConditionalGeneration
        if name == "videoauto_r1_qwen3":
            from lmms_eval.models.model_utils.qwen.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration as model_class
        model = model_class.from_pretrained(
            entry.path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(entry.path)
        trim_input_tokens = True
    elif name == "eagle25":
        model = AutoModel.from_pretrained(
            entry.path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(
            entry.path,
            trust_remote_code=True,
            use_fast=True,
        )
        processor.tokenizer.padding_side = "left"
        trim_input_tokens = False
    else:
        raise ValueError(f"No MLLM loader is implemented for {name!r}.")

    model.eval()
    return MLLMRuntime(
        name=name,
        model=model,
        processor=processor,
        device=device,
        trim_input_tokens=trim_input_tokens,
    )


def generate_mllm(
    runtime: MLLMRuntime,
    prompt: str,
    frames: list[Image.Image],
    media_type: MediaType,
    max_new_tokens: int = 1024,
) -> str:
    import torch

    if not frames:
        raise ValueError("At least one frame is required for MLLM generation.")
    if media_type == "image" and len(frames) != 1:
        raise ValueError("Image input requires exactly one frame.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": media_type},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    is_videoauto = runtime.name == "videoauto_r1_qwen3"
    if is_videoauto:
        from lmms_eval.models.simple.qwen3_vl_autothink import COT_SYSTEM_PROMPT_ANSWER_TWICE

        messages.insert(0, {"role": "system", "content": COT_SYSTEM_PROMPT_ANSWER_TWICE})
    chat_text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    processor_kwargs = {
        "text": [chat_text],
        "images": frames if media_type == "image" else None,
        "videos": [frames] if media_type == "video" else None,
        "padding": True,
        "return_tensors": "pt",
    }
    if runtime.name in ("qwen3_vl", "videoauto_r1_qwen3") and media_type == "video":
        processor_kwargs["do_sample_frames"] = False
    inputs = runtime.processor(**processor_kwargs).to(runtime.device)

    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if is_videoauto:
        generation_kwargs.update(
            max_new_tokens=4096,
            eos_token_id=runtime.processor.tokenizer.eos_token_id,
            pad_token_id=runtime.processor.tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    with torch.inference_mode():
        generated = runtime.model.generate(**inputs, **generation_kwargs)

    generated_ids = generated.sequences if is_videoauto else generated
    if runtime.trim_input_tokens:
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

    response = runtime.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if is_videoauto:
        from lmms_eval.models.simple.early_exit import compute_first_boxed_answer_probs
        from .parsing import extract_answer_letter

        confidence = compute_first_boxed_answer_probs(
            b=0, gen_ids=generated_ids[0], gen_out=generated, ans=response,
            task="v_oasis", tokenizer=runtime.processor.tokenizer,
        )
        response = response.split("<think>")[0] if confidence >= 0.98 else response.split("</think>")[-1]
        if not extract_answer_letter(response):
            raise ValueError("VideoAuto-R1 did not produce a valid selected answer.")
    return response


def generate_text(
    runtime: MLLMRuntime,
    prompt: str,
    max_new_tokens: int = 1024,
) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = runtime.processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(runtime.device)

    with torch.inference_mode():
        generated_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if runtime.trim_input_tokens:
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

    return runtime.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
