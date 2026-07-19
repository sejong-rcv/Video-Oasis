"""Shared runner for center-frame and frame-shuffle tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from tqdm import tqdm

from criteria.common.annotations import Annotation, load_annotations
from criteria.common.mllm import MediaType, generate_mllm, load_mllm
from criteria.common.outputs import (
    has_saved_prediction,
    prediction_path,
    save_prediction,
)
from criteria.common.prompts import build_mllm_video_prompt
from criteria.common.video_io import (
    frame_to_image,
    frames_to_images,
    open_video,
    segment_bounds,
    uniform_indices,
)


MLLMTemporalTest = Literal["center_frame", "frame_shuffle"]


@dataclass(frozen=True)
class SampledFrames:
    frames: list[Image.Image]
    indices: tuple[int, ...]
    media_type: MediaType


@dataclass(frozen=True)
class TemporalMLLMConfig:
    test: MLLMTemporalTest
    model: str
    anno_path: Path
    output_dir: Path
    device: str = "cuda"
    num_frames: int = 128
    max_new_tokens: int = 1024
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")


@dataclass(frozen=True)
class TemporalRunSummary:
    total: int
    completed: int
    skipped: int
    failed: int


def sample_center_frame(ann: Annotation) -> SampledFrames:
    reader = open_video(ann)
    start, end = segment_bounds(ann, reader)
    index = round((start + end) / 2)
    return SampledFrames(
        frames=[frame_to_image(reader, index)],
        indices=(index,),
        media_type="image",
    )


def sample_shuffled_frames(
    ann: Annotation,
    num_frames: int = 128,
) -> SampledFrames:
    reader = open_video(ann)
    start, end = segment_bounds(ann, reader)
    indices = uniform_indices(start, end, num_frames)

    rng = random.Random(f"{ann['db']}:{ann['qid']}")
    rng.shuffle(indices)

    return SampledFrames(
        frames=frames_to_images(reader, indices),
        indices=tuple(indices),
        media_type="video",
    )


def sample_frames(ann: Annotation, config: TemporalMLLMConfig) -> SampledFrames:
    if config.test == "center_frame":
        return sample_center_frame(ann)
    if config.test == "frame_shuffle":
        return sample_shuffled_frames(ann, config.num_frames)
    raise ValueError(f"Unknown temporal MLLM test: {config.test}")


def run_mllm_test(config: TemporalMLLMConfig) -> TemporalRunSummary:
    annotations = load_annotations(config.anno_path)
    runtime = load_mllm(config.model, config.device)

    completed = 0
    skipped = 0
    failed = 0

    for ann in tqdm(annotations, desc=f"{config.test}:{config.model}"):
        output_path = prediction_path(config.output_dir, config.model, ann)
        if not config.overwrite and has_saved_prediction(output_path):
            skipped += 1
            continue

        try:
            sampled = sample_frames(ann, config)
            prompt = build_mllm_video_prompt(ann, visual_word=sampled.media_type)
            prediction = generate_mllm(
                runtime=runtime,
                prompt=prompt,
                frames=sampled.frames,
                media_type=sampled.media_type,
                max_new_tokens=config.max_new_tokens,
            )
            save_prediction(
                output_path,
                ann,
                prediction,
                diagnostic={
                    "axis": "temporal",
                    "test": config.test,
                    "model": config.model,
                    "frame_indices": list(sampled.indices),
                    "num_frames": len(sampled.indices),
                },
            )
            completed += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"{output_path}: {exc}")

    summary = TemporalRunSummary(
        total=len(annotations),
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
    print(
        f"total={summary.total} completed={summary.completed} "
        f"skipped={summary.skipped} failed={summary.failed}"
    )
    return summary
