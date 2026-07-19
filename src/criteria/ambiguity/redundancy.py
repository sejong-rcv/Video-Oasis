"""Eagle2.5 chunk-wise inference for the Redundancy Test."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from criteria.common.annotations import Annotation, dump_json, load_annotations, sample_id
from criteria.common.mllm import generate_mllm, load_mllm
from criteria.common.outputs import prediction_path
from criteria.common.parsing import extract_answer_letter
from criteria.common.prompts import build_mllm_video_prompt
from criteria.common.video_io import frames_to_images, open_video, uniform_indices


@dataclass(frozen=True)
class RedundancyConfig:
    anno_path: Path
    output_dir: Path
    model: str = "eagle25"
    device: str = "cuda"
    num_chunks: int = 8
    frames_per_chunk: int = 16
    max_new_tokens: int = 1024
    overwrite: bool = False
    summarize_only: bool = False

    def __post_init__(self) -> None:
        if self.model != "eagle25":
            raise ValueError("The paper's Redundancy Test uses Eagle2.5 only.")
        if self.num_chunks < 1:
            raise ValueError("num_chunks must be at least 1")
        if self.frames_per_chunk < 1:
            raise ValueError("frames_per_chunk must be at least 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")


@dataclass(frozen=True)
class RedundancySummary:
    total: int
    completed: int
    skipped: int
    failed: int
    complete_results: int
    incomplete_results: int
    candidates: int


def chunk_frame_indices(
    start: int,
    end: int,
    num_chunks: int,
    frames_per_chunk: int,
) -> list[list[int]]:
    if end < start:
        raise ValueError("end must not precede start")
    edges = np.linspace(start, end + 1, num_chunks + 1, dtype=int)
    chunks = []
    for index in range(num_chunks):
        chunk_start = min(int(edges[index]), end)
        chunk_end = min(max(int(edges[index + 1]) - 1, chunk_start), end)
        chunks.append(uniform_indices(chunk_start, chunk_end, frames_per_chunk))
    return chunks


def load_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def state_matches_config(state: dict, config: RedundancyConfig) -> bool:
    diagnostic = state.get("diagnostic", {})
    return (
        diagnostic.get("test") == "redundancy"
        and diagnostic.get("model") == config.model
        and diagnostic.get("num_chunks") == config.num_chunks
        and diagnostic.get("frames_per_chunk") == config.frames_per_chunk
        and diagnostic.get("video_scope") == "full_video"
    )


def new_state(ann: Annotation, config: RedundancyConfig) -> dict:
    return {
        **dict(ann),
        "chunks": [],
        "complete": False,
        "all_chunks_correct": False,
        "diagnostic": {
            "axis": "ambiguity",
            "test": "redundancy",
            "model": config.model,
            "num_chunks": config.num_chunks,
            "frames_per_chunk": config.frames_per_chunk,
            "video_scope": "full_video",
        },
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)
        f.write("\n")
    temporary.replace(path)


def completed_chunk_indices(state: dict) -> set[int]:
    return {
        int(chunk["chunk_index"])
        for chunk in state.get("chunks", [])
        if isinstance(chunk, dict) and isinstance(chunk.get("chunk_index"), int)
    }


def process_annotation(
    ann: Annotation,
    config: RedundancyConfig,
    runtime,
) -> bool:
    output_path = prediction_path(config.output_dir, config.model, ann)
    state = None if config.overwrite else load_state(output_path)
    if state is not None and not state_matches_config(state, config):
        raise ValueError(f"Existing result uses a different configuration: {output_path}")
    if state is None:
        state = new_state(ann, config)
    if state.get("complete"):
        return False

    reader = open_video(ann)
    if len(reader) == 0:
        raise ValueError(f"Video has no frames: {ann['video_path']}")
    start, end = 0, len(reader) - 1
    chunks = chunk_frame_indices(start, end, config.num_chunks, config.frames_per_chunk)
    finished = completed_chunk_indices(state)
    prompt = build_mllm_video_prompt(ann)
    answer = str(ann.get("answer", "")).strip().upper()
    num_options = len(ann.get("options") or [])

    for chunk_index, indices in enumerate(chunks):
        if chunk_index in finished:
            continue
        prediction_raw = generate_mllm(
            runtime=runtime,
            prompt=prompt,
            frames=frames_to_images(reader, indices),
            media_type="video",
            max_new_tokens=config.max_new_tokens,
        )
        prediction = extract_answer_letter(prediction_raw, num_options)
        state["chunks"].append(
            {
                "chunk_index": chunk_index,
                "frame_indices": indices,
                "pred": prediction_raw,
                "parsed_prediction": prediction,
                "correct": bool(prediction) and prediction == answer,
            }
        )
        state["chunks"].sort(key=lambda item: item["chunk_index"])
        save_state(output_path, state)

    state["complete"] = len(completed_chunk_indices(state)) == config.num_chunks
    state["all_chunks_correct"] = state["complete"] and all(
        bool(chunk.get("correct")) for chunk in state["chunks"]
    )
    save_state(output_path, state)
    return True


def summarize_redundancy(
    annotations: list[Annotation],
    config: RedundancyConfig,
    completed: int,
    skipped: int,
    failed: int,
) -> RedundancySummary:
    candidates = []
    complete_results = 0

    for ann in annotations:
        path = prediction_path(config.output_dir, config.model, ann)
        state = load_state(path)
        if state is None or not state_matches_config(state, config) or not state.get("complete"):
            continue
        complete_results += 1
        if state.get("all_chunks_correct"):
            candidates.append({**dict(ann), "redundancy_result": str(path)})

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(candidates, config.output_dir / "redundancy_candidates.json")
    with (config.output_dir / "redundancy_candidate_ids.txt").open("w", encoding="utf-8") as f:
        for ann in candidates:
            f.write(f"{sample_id(ann)}\n")

    summary = RedundancySummary(
        total=len(annotations),
        completed=completed,
        skipped=skipped,
        failed=failed,
        complete_results=complete_results,
        incomplete_results=len(annotations) - complete_results,
        candidates=len(candidates),
    )
    dump_json(asdict(summary), config.output_dir / "summary.json")
    print(
        f"redundancy: total={summary.total} completed={summary.completed} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"candidates={summary.candidates}"
    )
    return summary


def run_redundancy(config: RedundancyConfig) -> RedundancySummary:
    annotations = load_annotations(config.anno_path)
    if config.summarize_only:
        return summarize_redundancy(annotations, config, 0, 0, 0)

    pending = []
    skipped = 0
    for ann in annotations:
        state = None if config.overwrite else load_state(
            prediction_path(config.output_dir, config.model, ann)
        )
        if state is not None and not state_matches_config(state, config):
            raise ValueError(
                "Existing Redundancy results use a different configuration. "
                "Use --overwrite or restore the original chunk settings."
            )
        if state is not None and state.get("complete"):
            skipped += 1
        else:
            pending.append(ann)

    runtime = load_mllm(config.model, config.device) if pending else None
    completed = 0
    failed = 0
    for ann in tqdm(pending, desc=f"redundancy:{config.model}"):
        try:
            completed += int(process_annotation(ann, config, runtime))
        except Exception as exc:
            failed += 1
            tqdm.write(f"{prediction_path(config.output_dir, config.model, ann)}: {exc}")

    return summarize_redundancy(
        annotations,
        config,
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
