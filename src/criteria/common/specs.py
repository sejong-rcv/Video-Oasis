"""Lightweight diagnostic test specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Axis = Literal["visual", "temporal", "ambiguity"]
Backend = Literal["llm", "mllm", "vlm", "postprocess", "manual"]


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str
    axis: Axis
    backend: Backend
    description: str
    paper_name: str
