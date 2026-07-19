"""Shared text-only runner for visual-dependency tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from criteria.common.annotations import load_annotations
from criteria.common.outputs import has_saved_prediction, prediction_path, save_prediction
from criteria.common.prompts import build_visual_dependency_prompt
from criteria.common.text import generate_text, load_text_model

from .contexts import ContextStore, VisualTest


@dataclass(frozen=True)
class VisualTextConfig:
    test: VisualTest
    model: str
    anno_path: Path
    output_dir: Path
    device: str = "cuda"
    transcript_dir: Path | None = None
    summary_file: Path | None = None
    max_new_tokens: int = 1024
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")


@dataclass(frozen=True)
class VisualRunSummary:
    total: int
    completed: int
    skipped: int
    missing_context: int
    failed: int


def run_visual_test(config: VisualTextConfig) -> VisualRunSummary:
    annotations = load_annotations(config.anno_path)
    contexts = ContextStore(
        test=config.test,
        transcript_dir=config.transcript_dir,
        summary_file=config.summary_file,
    )
    runtime = load_text_model(config.model, config.device)

    completed = 0
    skipped = 0
    missing_context = 0
    failed = 0

    for ann in tqdm(annotations, desc=f"{config.test}:{config.model}"):
        output_path = prediction_path(config.output_dir, config.model, ann)
        if not config.overwrite and has_saved_prediction(output_path):
            skipped += 1
            continue

        try:
            context = contexts.get(ann)
            if context is None:
                missing_context += 1
                continue
            prompt = build_visual_dependency_prompt(
                ann,
                context_label=context.label,
                context=context.text,
            )
            prediction = generate_text(
                runtime,
                prompt,
                max_new_tokens=config.max_new_tokens,
            )
            save_prediction(
                output_path,
                ann,
                prediction,
                diagnostic={
                    "axis": "visual",
                    "test": config.test,
                    "model": config.model,
                    "context_source": context.source,
                },
            )
            completed += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"{output_path}: {exc}")

    summary = VisualRunSummary(
        total=len(annotations),
        completed=completed,
        skipped=skipped,
        missing_context=missing_context,
        failed=failed,
    )
    print(
        f"total={summary.total} completed={summary.completed} "
        f"skipped={summary.skipped} missing_context={summary.missing_context} "
        f"failed={summary.failed}"
    )
    return summary
