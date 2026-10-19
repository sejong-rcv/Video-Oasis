"""Context lookup for visual-dependency tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from criteria.common.annotations import Annotation, resolve_video_path
from criteria.common.constants import DATA_ROOT


VisualTest = Literal["blind", "audio", "summary"]

DATASET_SLUGS = {
    "ImplicitQA": "implicitqa",
    "LVBench": "lvbench",
    "LongVideoBench": "longvideobench",
    "MINERVA": "minerva",
    "MLVU_Test": "mlvu_test",
    "MMR-V": "mmrvbench",
    "MVBench": "mvbench",
    "RTV-Bench": "rtv-bench",
    "TVBench": "tvbench",
    "VCR-Bench": "vcr-bench",
    "VSI-Bench": "vsi-bench",
    "Video-Holmes": "video-holmes",
    "Video-MME": "video-mme",
    "egoschema": "egoschema",
}


def audio_path(root: Path, ann: Annotation) -> Path:
    slug = DATASET_SLUGS[str(ann["db"])]
    video = resolve_video_path(str(ann["video_path"]))
    name = video.name
    if ann["db"] == "TVBench":
        name = str(video.relative_to(DATA_ROOT / "videos" / slug / "videos"))
    return root / slug / f"{name}.mp3"


def load_transcript(path: Path, ann: Annotation) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("transcript"), str):
        return None
    if data["transcript"].lstrip().startswith("Error:"):
        return None
    if "video_path" in data and data["video_path"] != ann["video_path"]:
        return None
    return data


def format_transcript_with_timestamps(data: dict) -> str:
    transcript = data["transcript"].strip()
    chunks = data.get("timestamps")
    if not chunks:
        return transcript
    lines = []
    try:
        for chunk in chunks:
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()
            if any(value is not None and (not math.isfinite(value) or value < 0)
                   for value in (start, end)):
                return transcript
            if start is not None and end is not None and end < start:
                return transcript
            start = f"{start:.1f}" if start is not None else "unknown"
            end = f"{end:.1f}" if end is not None else "unknown"
            if text:
                lines.append(f"[{start} - {end}] {text}")
    except (KeyError, TypeError, ValueError, AttributeError):
        return transcript
    return "\n".join(lines) or transcript


@dataclass(frozen=True)
class VisualContext:
    label: str
    text: str
    source: str | None = None


class ContextStore:
    def __init__(
        self,
        test: VisualTest,
        transcript_dir: Path | None = None,
        summary_file: Path | None = None,
    ) -> None:
        self.test = test
        self.transcript_dir = transcript_dir
        self.summary_file = summary_file
        self.summary_data = None

        if test == "audio" and transcript_dir is None:
            raise ValueError("Audio Test requires a transcript directory.")
        if test == "summary":
            if summary_file is None:
                raise ValueError("Summary Test requires a summary file.")
            with summary_file.open("r", encoding="utf-8") as f:
                self.summary_data = json.load(f)

    def get(self, ann: Annotation) -> VisualContext | None:
        if self.test == "blind":
            return VisualContext(label="Empty", text="")
        if self.test == "audio":
            return self._get_audio(ann)
        if self.test == "summary":
            return self._get_summary(ann)
        raise ValueError(f"Unknown visual test: {self.test}")

    def _get_audio(self, ann: Annotation) -> VisualContext | None:
        slug = DATASET_SLUGS.get(str(ann["db"]))
        if slug is None or self.transcript_dir is None:
            return None

        path = audio_path(self.transcript_dir, ann).with_suffix(".mp3.json")
        candidates = [path]
        if ann["db"] != "TVBench" and ann.get("video"):
            name = Path(str(ann["video"])).name
            candidates.append(self.transcript_dir / slug / f"{name}.mp3.json")
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return None

        data = load_transcript(path, ann)
        if data is None:
            return None
        if not data["transcript"].strip():
            return None
        transcript = format_transcript_with_timestamps(data)
        return VisualContext(label="Audio Transcript", text=transcript, source=str(path))

    def _get_summary(self, ann: Annotation) -> VisualContext | None:
        if self.summary_data is None:
            return None
        key = str(ann["video_path"]).replace("../data/benchmarks", "")
        item = self.summary_data.get(key)
        if not item:
            return None
        summaries = [str(value).strip() for value in item.get("summary", []) if str(value).strip()]
        if not summaries:
            return None
        return VisualContext(
            label="Summary",
            text="\n".join(summaries),
            source=str(self.summary_file),
        )
