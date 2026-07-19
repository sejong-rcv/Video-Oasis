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
    elif name == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
    inputs = runtime.processor(**processor_kwargs).to(runtime.device)

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
    )[0]
