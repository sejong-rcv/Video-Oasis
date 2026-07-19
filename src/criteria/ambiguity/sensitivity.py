"""Build a manual-review queue for Frame Shuffling sensitivity checks."""

from __future__ import annotations

import json
from pathlib import Path

from criteria.common.annotations import Annotation, dump_json, sample_id


def build_sensitivity_queue(
    annotations: list[Annotation],
    diagnostic_report: Path,
    output_dir: Path,
) -> int:
    with diagnostic_report.open("r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise ValueError(f"Expected a JSON list: {diagnostic_report}")

    annotations_by_id = {sample_id(ann): ann for ann in annotations}
    queue = []
    for item in results:
        sid = str(item.get("sample_id", ""))
        frame_shuffle = item.get("tests", {}).get("frame_shuffle", {})
        if frame_shuffle.get("status") != "positive":
            continue
        ann = annotations_by_id.get(sid)
        if ann is None:
            continue
        queue.append(
            {
                **dict(ann),
                "sample_id": sid,
                "frame_shuffle": frame_shuffle,
                "manual_decision": "",
                "manual_note": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(queue, output_dir / "sensitivity_candidates.json")
    with (output_dir / "sensitivity_candidate_ids.txt").open("w", encoding="utf-8") as f:
        for item in queue:
            f.write(f"{item['sample_id']}\n")
    print(f"sensitivity: candidates={len(queue)}")
    return len(queue)
