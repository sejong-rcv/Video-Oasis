"""Output-file conventions for diagnostic predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .annotations import Annotation, sample_id, video_interval


def prediction_path(output_dir: str | Path, model_name: str, ann: Annotation) -> Path:
    return Path(output_dir) / model_name / f"{sample_id(ann)}.json"


def matches_video_input(payload: dict, ann: Annotation) -> bool:
    if payload.get("video_path") != ann.get("video_path"):
        return False
    try:
        interval = video_interval(ann)
        if video_interval(payload) != interval:
            return False
    except (TypeError, ValueError):
        return False
    if interval is None:
        return True
    diagnostic = payload.get("diagnostic")
    return isinstance(diagnostic, dict) and diagnostic.get("video_scope") == "segment"


def load_saved_prediction(
    path: str | Path,
    ann: Annotation | None = None,
) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    prediction = data.get("pred")
    if not isinstance(prediction, str) or not prediction.strip():
        return None
    if prediction.strip().lower().startswith("error::"):
        return None
    if ann is not None and not matches_video_input(data, ann):
        return None
    return data


def has_saved_prediction(path: str | Path, ann: Annotation | None = None) -> bool:
    return load_saved_prediction(path, ann) is not None


def save_prediction(
    path: str | Path,
    ann: Annotation,
    pred: str,
    diagnostic: dict[str, Any] | None = None,
) -> None:
    payload = dict(ann)
    payload["pred"] = pred
    if diagnostic is not None:
        payload["diagnostic"] = diagnostic
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")
