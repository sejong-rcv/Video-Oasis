"""Shared video loading and frame-access helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from .annotations import Annotation, resolve_video_path


def open_video(ann: Annotation) -> VideoReader:
    return VideoReader(str(resolve_video_path(ann["video_path"])), ctx=cpu(0), num_threads=1)


def segment_bounds(ann: Annotation, reader: VideoReader) -> tuple[int, int]:
    total_frames = len(reader)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {ann['video_path']}")

    if ann.get("db") == "RTV-Bench":
        fps = float(reader.get_avg_fps())
        start = round(float(ann["start_time"]) * fps)
        end = round(float(ann["end_time"]) * fps)
    else:
        start = 0
        end = total_frames - 1

    start = max(0, min(start, total_frames - 1))
    end = max(start, min(end, total_frames - 1))
    return start, end


def frame_to_image(reader: VideoReader, index: int) -> Image.Image:
    return Image.fromarray(reader[index].asnumpy()).convert("RGB")


def uniform_indices(start: int, end: int, num_frames: int) -> list[int]:
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    return np.linspace(start, end, num_frames, dtype=int).tolist()


def frames_to_images(reader: VideoReader, indices: Sequence[int]) -> list[Image.Image]:
    frames = reader.get_batch(list(indices)).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]
