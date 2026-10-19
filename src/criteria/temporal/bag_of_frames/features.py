"""Video-frame feature extraction for Bag-of-Frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from criteria.common.annotations import Annotation, video_interval
from criteria.common.video_io import frames_to_images, open_video, segment_bounds
from criteria.common.vlm import encode_images


@dataclass(frozen=True)
class FeatureExtractionSummary:
    total: int
    completed: int
    skipped: int
    failed: int


def video_name(ann: Annotation) -> str:
    path = Path(ann["video_path"])
    if ann["db"] == "TVBench":
        return f"{path.parent.name}_{path.name}"
    return path.name


def feature_path(feature_dir: str | Path, ann: Annotation) -> Path:
    name = video_name(ann)
    interval = video_interval(ann)
    if interval is not None:
        start, end = interval
        name = f"{name}__{start}_{end}s"
    return Path(feature_dir) / f"{ann['db']}**@@**{name}.pt"


def model_feature_dir(feature_root: str | Path, model_family: str) -> Path:
    feature_root = Path(feature_root)
    if feature_root.name == model_family:
        return feature_root
    return feature_root / model_family


def feature_candidates(
    feature_dir: Path,
    ann: Annotation,
    model_family: str | None,
) -> list[Path]:
    root = model_feature_dir(feature_dir, model_family) if model_family else feature_dir
    if video_interval(ann) is not None:
        return [feature_path(root, ann)]

    db = str(ann["db"])
    name = video_name(ann)
    return [
        feature_path(root, ann),
        root / f"{db}_@@_{name}.pt",
        root / f"{db.lower()}_@@_{name}.pt",
        root / f"{name}.pt",
    ]


def resolve_feature_path(
    feature_dir: Path,
    ann: Annotation,
    model_family: str | None,
) -> Path | None:
    return next(
        (
            path
            for path in feature_candidates(feature_dir, ann, model_family)
            if path.is_file()
        ),
        None,
    )


def load_feature_tensor(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        payload = payload.get("features")
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected a feature tensor in {path}")
    return payload


def unique_videos(annotations: list[Annotation]) -> list[Annotation]:
    videos = {}
    for ann in annotations:
        key = (str(ann["db"]), str(ann["video_path"]), video_interval(ann))
        videos.setdefault(key, ann)
    return list(videos.values())


def sampled_video_indices(
    reader,
    max_frames: int,
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    if len(reader) == 0:
        raise ValueError("Video has no frames.")

    end = len(reader) - 1 if end is None else end
    frame_step = max(1, int(reader.get_avg_fps()))
    indices = np.arange(start, end + 1, frame_step, dtype=int)
    if len(indices) > max_frames:
        indices = np.linspace(start, end, max_frames, dtype=int)
    return np.clip(indices, start, end).tolist()


def extract_video_features(
    runtime,
    ann: Annotation,
    batch_size: int,
    max_frames: int,
):
    reader = open_video(ann)
    frame_start, frame_end = segment_bounds(ann, reader)
    indices = sampled_video_indices(reader, max_frames, frame_start, frame_end)
    batches = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = frames_to_images(reader, batch_indices)
        batches.append(encode_images(runtime, images).cpu())

    if not batches:
        raise ValueError(f"No features extracted from {ann['video_path']}")
    return torch.cat(batches, dim=0)


def extract_features(
    runtime,
    annotations: list[Annotation],
    feature_dir: Path,
    batch_size: int,
    max_frames: int,
    overwrite: bool,
) -> FeatureExtractionSummary:
    videos = unique_videos(annotations)
    if not runtime.entry.family:
        raise ValueError(f"Model family is not configured for {runtime.entry.name!r}")
    model_dir = model_feature_dir(feature_dir, runtime.entry.family)
    model_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    for ann in tqdm(videos, desc=f"extract:{runtime.entry.name}"):
        output_path = feature_path(model_dir, ann)
        if output_path.is_file() and not overwrite:
            skipped += 1
            continue

        try:
            features = extract_video_features(
                runtime,
                ann,
                batch_size=batch_size,
                max_frames=max_frames,
            )
            torch.save(features, output_path)
            completed += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"{output_path}: {exc}")

    summary = FeatureExtractionSummary(
        total=len(videos),
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
    print(
        f"extraction: total={summary.total} completed={summary.completed} "
        f"skipped={summary.skipped} failed={summary.failed}"
    )
    return summary
