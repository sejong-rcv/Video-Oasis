"""Shared model metadata for diagnostic tests."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import REPO_ROOT


@dataclass(frozen=True)
class ModelEntry:
    name: str
    backend: str
    path: str
    family: str | None = None


MODEL_REGISTRY = {
    "qwen25_vl": ModelEntry("qwen25_vl", "mllm", str(REPO_ROOT / "data/models/Qwen2.5-VL-7B-Instruct")),
    "qwen3_vl": ModelEntry("qwen3_vl", "mllm", str(REPO_ROOT / "data/models/Qwen3-VL-8B-Instruct")),
    "eagle25": ModelEntry("eagle25", "mllm", str(REPO_ROOT / "data/models/Eagle2.5-8B")),
    "clip-vit-l-14": ModelEntry(
        "clip-vit-l-14",
        "vlm",
        "openai/clip-vit-large-patch14",
        family="clip",
    ),
    "eva-clip-8b": ModelEntry(
        "eva-clip-8b",
        "vlm",
        "BAAI/EVA-CLIP-8B",
        family="eva",
    ),
    "longclip": ModelEntry(
        "longclip",
        "vlm",
        str(REPO_ROOT / "src/criteria/common/longclip/longclip-L.pt"),
        family="longclip",
    ),
}


def get_model_entry(name: str) -> ModelEntry:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc


def model_names(backend: str | None = None) -> tuple[str, ...]:
    names = (
        name
        for name, entry in MODEL_REGISTRY.items()
        if backend is None or entry.backend == backend
    )
    return tuple(sorted(names))
