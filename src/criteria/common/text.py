"""Text-only generation shared by visual-dependency tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mllm import load_mllm
from .models import model_names


@dataclass
class TextRuntime:
    name: str
    model: Any
    processor: Any
    device: str
    trim_input_tokens: bool = True


def text_model_names() -> tuple[str, ...]:
    return model_names("mllm")


def load_text_model(name: str, device: str = "cuda") -> TextRuntime:
    runtime = load_mllm(name, device)
    return TextRuntime(
        name=name,
        model=runtime.model,
        processor=runtime.processor,
        device=device,
        trim_input_tokens=runtime.trim_input_tokens,
    )


def generate_text(
    runtime: TextRuntime,
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
