"""Evaluation runner for Bag-of-Frames predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from criteria.common.annotations import Annotation, load_annotations
from criteria.common.outputs import (
    load_saved_prediction,
    prediction_path,
    save_prediction,
)
from criteria.common.vlm import encode_texts, load_vlm

from .bag_of_frames.features import (
    FeatureExtractionSummary,
    extract_features,
    load_feature_tensor,
    resolve_feature_path,
)
from .bag_of_frames.scoring import Aggregation, ScoringConfig, score_bag_of_frames


@dataclass(frozen=True)
class BagOfFramesConfig:
    model: str
    anno_path: Path
    feature_dir: Path
    output_dir: Path
    device: str = "cuda"
    batch_size: int = 32
    max_frames: int = 2048
    top_k: int = 32
    aggregation: Aggregation = "mean"
    max_text_length: int = 64
    overwrite: bool = False

    def __post_init__(self) -> None:
        ScoringConfig(top_k=self.top_k, aggregation=self.aggregation)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        if self.max_text_length < 1:
            raise ValueError("max_text_length must be at least 1")


@dataclass(frozen=True)
class BagOfFramesEvaluationSummary:
    total: int
    completed: int
    skipped: int
    missing_features: int
    failed: int
    correct: int

    @property
    def evaluated(self) -> int:
        return self.completed + self.skipped

    @property
    def accuracy(self) -> float:
        return self.correct / self.evaluated if self.evaluated else 0.0


@dataclass(frozen=True)
class BagOfFramesRunSummary:
    extraction: FeatureExtractionSummary
    evaluation: BagOfFramesEvaluationSummary


def answer_letter(index: int) -> str:
    return chr(ord("A") + index)


def evaluate_bag_of_frames(
    config: BagOfFramesConfig,
    annotations: list[Annotation],
    runtime,
) -> BagOfFramesEvaluationSummary:
    scoring_config = ScoringConfig(
        top_k=config.top_k,
        aggregation=config.aggregation,
    )
    run_name = f"{config.model}_k{config.top_k}_{config.aggregation}"

    completed = 0
    skipped = 0
    missing_features = 0
    failed = 0
    correct = 0

    for ann in tqdm(annotations, desc=f"bof:{run_name}"):
        output_path = prediction_path(config.output_dir, run_name, ann)
        saved = None if config.overwrite else load_saved_prediction(output_path)
        if saved is not None:
            skipped += 1
            if saved["pred"] == ann.get("answer"):
                correct += 1
            continue

        path = resolve_feature_path(
            config.feature_dir,
            ann,
            runtime.entry.family,
        )
        if path is None:
            missing_features += 1
            continue

        try:
            options = ann.get("options")
            if not options:
                raise ValueError("The sample has no answer options.")

            text_features = encode_texts(
                runtime,
                [ann["question"], *options],
                max_length=config.max_text_length,
            )
            result = score_bag_of_frames(
                question_features=text_features[:1],
                option_features=text_features[1:],
                video_features=load_feature_tensor(path),
                config=scoring_config,
            )
            prediction = answer_letter(result.answer_index)
            save_prediction(
                output_path,
                ann,
                prediction,
                diagnostic={
                    "axis": "temporal",
                    "test": "bag_of_frames",
                    "model": config.model,
                    "top_k": config.top_k,
                    "aggregation": config.aggregation,
                    "selected_feature_indices": result.selected_indices,
                    "retrieval_scores": result.retrieval_scores,
                    "option_scores": result.scores,
                    "feature_path": str(path),
                },
            )
            completed += 1
            if prediction == ann.get("answer"):
                correct += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"{output_path}: {exc}")

    summary = BagOfFramesEvaluationSummary(
        total=len(annotations),
        completed=completed,
        skipped=skipped,
        missing_features=missing_features,
        failed=failed,
        correct=correct,
    )
    print(
        f"evaluation: total={summary.total} completed={summary.completed} "
        f"skipped={summary.skipped} missing_features={summary.missing_features} "
        f"failed={summary.failed} accuracy={summary.accuracy:.4f}"
    )
    return summary


def run_bag_of_frames(config: BagOfFramesConfig) -> BagOfFramesRunSummary:
    annotations = load_annotations(config.anno_path)
    runtime = load_vlm(config.model, config.device)
    extraction = extract_features(
        runtime=runtime,
        annotations=annotations,
        feature_dir=config.feature_dir,
        batch_size=config.batch_size,
        max_frames=config.max_frames,
        overwrite=config.overwrite,
    )
    evaluation = evaluate_bag_of_frames(config, annotations, runtime)
    return BagOfFramesRunSummary(
        extraction=extraction,
        evaluation=evaluation,
    )
