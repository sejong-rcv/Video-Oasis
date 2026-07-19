"""Context lookup for visual-dependency tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from criteria.common.annotations import Annotation


VisualTest = Literal["blind", "audio", "summary"]

DATASET_SLUGS = {
    "LongVideoBench": "longvideobench",
    "MLVU_Test": "mlvu_test",
    "MMR-V": "mmrvbench",
    "MVBench": "mvbench",
    "RTV-Bench": "rtv-bench",
    "TVBench": "tvbench",
    "VCR-Bench": "vcr-bench",
    "Video-Holmes": "video-holmes",
    "Video-MME": "video-mme",
}


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

        names = {
            Path(str(ann["video_path"])).name,
            Path(str(ann.get("video", ""))).name,
        }
        candidates = [
            self.transcript_dir / slug / f"{name}.mp3.json"
            for name in names
            if name
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return None

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        transcript = str(data.get("transcript", "")).strip()
        if not transcript:
            return None
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
