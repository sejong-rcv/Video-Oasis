"""Cross-model disagreement analysis for the Consistency Test."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from criteria.common.annotations import Annotation, dump_json, sample_id
from criteria.common.parsing import extract_answer_letter


ConsistencyRule = Literal[
    "maximal-disagreement",
    "option-coverage",
    "no-majority",
    "all-unique",
]


@dataclass(frozen=True)
class PredictionSource:
    model: str
    path: Path


@dataclass(frozen=True)
class ConsistencySummary:
    total: int
    complete: int
    incomplete: int
    no_majority: int
    all_unique: int
    option_coverage: int
    maximal_disagreement: int
    candidates: int


def parse_prediction_source(value: str) -> PredictionSource:
    model, separator, path = value.partition("=")
    if not separator or not model.strip() or not path.strip():
        raise ValueError(f"Expected MODEL=PREDICTION_PATH, got {value!r}")
    return PredictionSource(model=model.strip(), path=Path(path.strip()))


def nested_score(payload: dict) -> dict:
    for key, value in payload.items():
        if key.endswith("_score") and isinstance(value, dict):
            return value
    return {}


def prediction_sample_id(
    payload: dict,
    annotations: list[Annotation],
) -> str | None:
    score = nested_score(payload)
    db = score.get("task_type") or score.get("db") or payload.get("db")
    qid = score.get("question_id") or score.get("qid") or payload.get("qid")
    if db is not None and qid is not None:
        return sample_id({"db": db, "qid": qid})

    doc_id = payload.get("doc_id")
    if isinstance(doc_id, int) and 0 <= doc_id < len(annotations):
        return sample_id(annotations[doc_id])
    return None


def raw_prediction(payload: dict) -> object:
    score = nested_score(payload)
    if score.get("pred_answer") not in (None, ""):
        return score["pred_answer"]

    filtered = payload.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return filtered[0]

    responses = payload.get("resps")
    if isinstance(responses, list) and responses:
        first = responses[0]
        if isinstance(first, list) and first:
            return first[0]
        return first
    return payload.get("pred", "")


def load_prediction_payloads(path: Path) -> tuple[list[tuple[int, dict]], list[dict]]:
    payloads = []
    errors = []
    with path.open("r", encoding="utf-8") as f:
        first_character = f.read(1)
        f.seek(0)
        if first_character == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list: {path}")
            payloads.extend(enumerate(data, start=1))
            return payloads, errors

        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payloads.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                errors.append(
                    {"line": line_number, "status": "invalid_json", "error": str(exc)}
                )
    return payloads, errors


def load_model_predictions(
    source: PredictionSource,
    annotations: list[Annotation],
    option_counts: dict[str, int],
) -> tuple[dict[str, str], list[dict]]:
    predictions: dict[str, str] = {}
    payloads, errors = load_prediction_payloads(source.path)
    conflicting_ids = set()

    for line_number, payload in payloads:
        sid = prediction_sample_id(payload, annotations)
        if sid is None or sid not in option_counts:
            errors.append({"line": line_number, "status": "unknown_sample"})
            continue
        if sid in conflicting_ids:
            continue
        prediction = extract_answer_letter(raw_prediction(payload), option_counts[sid])
        if not prediction:
            errors.append({"line": line_number, "sample_id": sid, "status": "parse_error"})
            continue
        if sid in predictions and predictions[sid] != prediction:
            errors.append(
                {
                    "line": line_number,
                    "sample_id": sid,
                    "status": "conflicting_duplicate",
                    "previous": predictions[sid],
                    "current": prediction,
                }
            )
            predictions.pop(sid, None)
            conflicting_ids.add(sid)
            continue
        predictions[sid] = prediction

    return predictions, errors


def run_consistency(
    annotations: list[Annotation],
    sources: list[PredictionSource],
    output_dir: Path,
    rule: ConsistencyRule = "maximal-disagreement",
) -> ConsistencySummary:
    if len(sources) != 5:
        raise ValueError(f"Consistency Test requires exactly 5 models, got {len(sources)}")
    model_names = [source.model for source in sources]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Consistency model names must be unique.")

    option_counts = {sample_id(ann): len(ann.get("options") or []) for ann in annotations}
    predictions_by_model = {}
    source_errors = {}
    for source in sources:
        predictions, errors = load_model_predictions(source, annotations, option_counts)
        predictions_by_model[source.model] = predictions
        source_errors[source.model] = errors

    results = []
    candidates = []
    incomplete = 0
    no_majority_count = 0
    all_unique_count = 0
    option_coverage_count = 0
    maximal_disagreement_count = 0

    for ann in annotations:
        sid = sample_id(ann)
        model_predictions = {
            model: predictions_by_model[model].get(sid, "") for model in model_names
        }
        complete = all(model_predictions.values())
        votes = Counter(model_predictions.values()) if complete else Counter()
        no_majority = complete and max(votes.values()) <= 2
        all_unique = complete and len(votes) == len(model_names)
        allowed_options = {
            chr(ord("A") + index) for index in range(option_counts[sid])
        }
        option_coverage = complete and set(votes) == allowed_options
        maximal_disagreement = complete and len(votes) == min(
            option_counts[sid], len(model_names)
        )
        candidate = {
            "maximal-disagreement": maximal_disagreement,
            "option-coverage": option_coverage,
            "no-majority": no_majority,
            "all-unique": all_unique,
        }[rule]

        incomplete += int(not complete)
        no_majority_count += int(no_majority)
        all_unique_count += int(all_unique)
        option_coverage_count += int(option_coverage)
        maximal_disagreement_count += int(maximal_disagreement)

        result = {
            "sample_id": sid,
            "db": ann.get("db"),
            "qid": ann.get("qid"),
            "num_options": option_counts[sid],
            "predictions": model_predictions,
            "votes": dict(sorted(votes.items())),
            "complete": complete,
            "no_majority": no_majority,
            "all_unique": all_unique,
            "option_coverage": option_coverage,
            "maximal_disagreement": maximal_disagreement,
            "candidate": candidate,
        }
        results.append(result)
        if candidate:
            candidates.append({**dict(ann), "consistency": result})

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(results, output_dir / "consistency_results.json")
    dump_json(candidates, output_dir / "consistency_candidates.json")
    dump_json(source_errors, output_dir / "consistency_source_errors.json")
    write_ids(
        output_dir / "consistency_candidate_ids.txt",
        [item["sample_id"] for item in results if item["candidate"]],
    )

    summary = ConsistencySummary(
        total=len(annotations),
        complete=len(annotations) - incomplete,
        incomplete=incomplete,
        no_majority=no_majority_count,
        all_unique=all_unique_count,
        option_coverage=option_coverage_count,
        maximal_disagreement=maximal_disagreement_count,
        candidates=len(candidates),
    )
    dump_json(
        {**summary.__dict__, "rule": rule, "models": model_names},
        output_dir / "summary.json",
    )
    print(
        f"consistency: total={summary.total} complete={summary.complete} "
        f"incomplete={summary.incomplete} candidates={summary.candidates}"
    )
    return summary


def write_ids(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"{sid}\n")
