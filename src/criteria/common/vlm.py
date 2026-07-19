"""Shared loading and encoding helpers for static vision-language models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .models import ModelEntry, get_model_entry


LONGCLIP_REPO_ID = "BeichenZhang/LongCLIP-L"
LONGCLIP_FILENAME = "longclip-L.pt"


@dataclass
class VLMRuntime:
    entry: ModelEntry
    model: Any
    processor: Any
    tokenizer: Any
    device: str


def ensure_longclip_checkpoint(checkpoint: str | Path, downloader=None) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        return checkpoint

    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        downloader(
            repo_id=LONGCLIP_REPO_ID,
            filename=LONGCLIP_FILENAME,
            revision="main",
            local_dir=str(checkpoint.parent),
        )
    )
    if not downloaded.is_file():
        raise FileNotFoundError(
            f"LongCLIP download completed without a checkpoint: {downloaded}"
        )
    return downloaded


def load_vlm(name: str, device: str = "cuda") -> VLMRuntime:
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    entry = get_model_entry(name)
    if entry.backend != "vlm":
        raise ValueError(f"Model {name!r} is not a VLM (backend={entry.backend!r}).")

    if entry.family == "longclip":
        checkpoint = ensure_longclip_checkpoint(entry.path)

        try:
            from .longclip.longclip import load as load_longclip
            from .longclip.longclip import tokenize as tokenize_longclip
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LongCLIP dependencies are unavailable; install ftfy and the "
                "requirements used by criteria.common.longclip."
            ) from exc

        model, processor = load_longclip(str(checkpoint), device=device)
        model = model.to(device)
        tokenizer = tokenize_longclip
    else:
        model_kwargs: dict[str, Any] = {}
        if entry.family == "eva":
            model_kwargs.update(torch_dtype=torch.float16, trust_remote_code=True)

        model = AutoModel.from_pretrained(entry.path, **model_kwargs).to(device)
        processor_path = (
            "openai/clip-vit-large-patch14"
            if entry.family == "eva"
            else entry.path
        )
        processor = AutoProcessor.from_pretrained(processor_path)
        if entry.family == "eva":
            tokenizer = processor
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(entry.path)
            except (OSError, ValueError):
                tokenizer = processor

    model.eval()
    return VLMRuntime(
        entry=entry,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
    )


def encode_texts(
    runtime: VLMRuntime,
    texts: list[str],
    max_length: int = 64,
):
    import torch
    import torch.nn.functional as F

    if not texts:
        raise ValueError("At least one text is required.")

    if runtime.entry.family == "longclip":
        inputs = runtime.tokenizer(texts).to(runtime.device)
        with torch.inference_mode():
            features = runtime.model.encode_text(inputs)
    else:
        inputs = runtime.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
        with torch.inference_mode():
            if hasattr(runtime.model, "get_text_features"):
                features = runtime.model.get_text_features(**inputs)
            elif hasattr(runtime.model, "encode_text"):
                features = runtime.model.encode_text(**inputs)
            else:
                raise TypeError(
                    f"{runtime.entry.name} does not expose a text encoder."
                )

    return F.normalize(features, dim=-1)


def encode_images(runtime: VLMRuntime, images: list[Image.Image]):
    import torch
    import torch.nn.functional as F

    if not images:
        raise ValueError("At least one image is required.")

    with torch.inference_mode():
        if runtime.entry.family == "longclip":
            pixel_values = torch.stack(
                [runtime.processor(image) for image in images]
            ).to(runtime.device)
            features = runtime.model.encode_image(pixel_values)
        else:
            inputs = runtime.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )
            model_dtype = next(runtime.model.parameters()).dtype
            inputs = {
                key: value.to(
                    runtime.device,
                    dtype=model_dtype if value.is_floating_point() else None,
                )
                for key, value in inputs.items()
            }
            if hasattr(runtime.model, "get_image_features"):
                features = runtime.model.get_image_features(**inputs)
            elif hasattr(runtime.model, "encode_image"):
                features = runtime.model.encode_image(inputs["pixel_values"])
            else:
                raise TypeError(
                    f"{runtime.entry.name} does not expose an image encoder."
                )

    return F.normalize(features, dim=-1)
