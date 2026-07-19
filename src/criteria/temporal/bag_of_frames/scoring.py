"""Top-k frame retrieval and answer-option scoring for Bag-of-Frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Aggregation = Literal["mean", "max"]


@dataclass(frozen=True)
class ScoringConfig:
    top_k: int = 32
    aggregation: Aggregation = "mean"

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.aggregation not in ("mean", "max"):
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


@dataclass(frozen=True)
class ScoringResult:
    answer_index: int
    scores: list[float]
    selected_indices: list[int]
    retrieval_scores: list[float]


def score_bag_of_frames(
    question_features: torch.Tensor,
    option_features: torch.Tensor,
    video_features: torch.Tensor,
    config: ScoringConfig,
) -> ScoringResult:
    if video_features.ndim != 2 or len(video_features) == 0:
        raise ValueError("video_features must have shape (frames, feature_dim)")
    if question_features.ndim == 1:
        question_features = question_features.unsqueeze(0)
    if question_features.shape[0] != 1:
        raise ValueError("Exactly one question feature is required")
    if option_features.ndim != 2 or len(option_features) == 0:
        raise ValueError("At least one option feature is required")

    device = question_features.device
    dtype = question_features.dtype
    question_features = F.normalize(question_features, dim=-1)
    option_features = F.normalize(option_features.to(device=device, dtype=dtype), dim=-1)
    video_features = F.normalize(video_features.to(device=device, dtype=dtype), dim=-1)

    retrieval = (question_features @ video_features.T).squeeze(0)
    actual_k = min(config.top_k, len(video_features))
    retrieval_scores, selected_indices = torch.topk(retrieval, actual_k)
    selected_features = video_features[selected_indices]

    if config.aggregation == "mean":
        pooled = F.normalize(selected_features.mean(dim=0), dim=-1)
        answer_scores = pooled.unsqueeze(0) @ option_features.T
        answer_scores = answer_scores.squeeze(0)
    else:
        frame_option_scores = selected_features @ option_features.T
        answer_scores = frame_option_scores.max(dim=0).values

    return ScoringResult(
        answer_index=int(answer_scores.argmax().item()),
        scores=answer_scores.detach().float().cpu().tolist(),
        selected_indices=selected_indices.detach().cpu().tolist(),
        retrieval_scores=retrieval_scores.detach().float().cpu().tolist(),
    )
