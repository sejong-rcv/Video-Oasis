"""Annotation loading and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import DATA_ROOT


Annotation = dict[str, Any]


def load_annotations(path: str | Path) -> list[Annotation]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def sample_id(ann: Annotation) -> str:
    return f"{ann['db']}**@@**{str(ann['qid']).replace('/', '_')}"


def resolve_video_path(video_path: str) -> Path:
    if video_path.startswith("../data/benchmarks"):
        rel_path = video_path.removeprefix("../data/benchmarks").lstrip("/")
        return DATA_ROOT / rel_path
    return Path(video_path)
