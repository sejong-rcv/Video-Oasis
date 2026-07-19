"""Output-file conventions for diagnostic predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .annotations import Annotation, sample_id


def prediction_path(output_dir: str | Path, model_name: str, ann: Annotation) -> Path:
    return Path(output_dir) / model_name / f"{sample_id(ann)}.json"


def load_saved_prediction(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except Exception:
        return None
    return data if data.get("pred") else None


def has_saved_prediction(path: str | Path) -> bool:
    return load_saved_prediction(path) is not None


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
