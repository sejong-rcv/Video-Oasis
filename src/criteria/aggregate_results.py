"""Aggregate diagnostic predictions into consensus shortcut decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from criteria.common.annotations import Annotation, dump_json, load_annotations, sample_id
from criteria.common.constants import DATA_ROOT, DEFAULT_ANNO_PATH
from criteria.common.parsing import extract_answer_letter
from criteria.visual.contexts import ContextStore


CRITERIA_ROOT = Path(__file__).resolve().parent
DEFAULT_VISUAL_OUTPUT_ROOT = CRITERIA_ROOT / "visual" / "output"
DEFAULT_TEMPORAL_OUTPUT_ROOT = CRITERIA_ROOT / "temporal" / "output"
DEFAULT_REPORT_DIR = CRITERIA_ROOT / "reports"

TestStatus = Literal["positive", "negative", "incomplete", "not_applicable", "restored"]


@dataclass(frozen=True)
class TestConfig:
    name: str
    output_dir: Path
    model_dirs: dict[str, str]
    requires_audio: bool = False


def build_test_configs(
    visual_root: Path,
    temporal_root: Path,
    top_k: int,
    aggregation: str,
) -> tuple[TestConfig, ...]:
    mllms = {name: name for name in ("qwen25_vl", "qwen3_vl", "eagle25")}
    vlms = {
        name: f"{name}_k{top_k}_{aggregation}"
        for name in ("clip-vit-l-14", "eva-clip-8b", "longclip")
    }
    return (
        TestConfig("blind", visual_root / "blind", mllms),
        TestConfig("audio", visual_root / "audio", mllms, requires_audio=True),
        TestConfig("summary", visual_root / "summary", mllms),
        TestConfig("center_frame", temporal_root / "center_frame", mllms),
        TestConfig("frame_shuffle", temporal_root / "frame_shuffle", mllms),
        TestConfig("bag_of_frames", temporal_root / "bag_of_frames", vlms),
    )


def load_restore_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def normalize_answer(answer: object, num_options: int) -> str:
    text = str(answer).strip().upper()
    if len(text) == 1 and "A" <= text <= "N":
        return text if ord(text) - ord("A") < num_options else ""
    return extract_answer_letter(text, num_options)


def read_prediction(path: Path, ground_truth: str, num_options: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "status": "missing",
        "prediction": "",
        "correct": False,
    }
    if not path.is_file():
        return result

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        result.update(status="invalid_json", error=str(exc))
        return result

    raw_prediction = payload.get("pred", "")
    prediction = extract_answer_letter(raw_prediction, num_options)
    if not prediction:
        result.update(status="parse_error", raw_prediction=str(raw_prediction))
        return result

    result.update(
        status="ok",
        prediction=prediction,
        correct=prediction == ground_truth,
    )
    return result


def evaluate_test(
    ann: Annotation,
    config: TestConfig,
    consensus: int,
    applicable: bool,
    restored: bool,
) -> dict[str, Any]:
    if not applicable:
        return {
            "status": "not_applicable",
            "correct_models": 0,
            "models": {},
        }

    sid = sample_id(ann)
    options = ann.get("options") or []
    ground_truth = normalize_answer(ann.get("answer", ""), len(options))
    models = {
        model: read_prediction(
            config.output_dir / model_dir / f"{sid}.json",
            ground_truth,
            len(options),
        )
        for model, model_dir in config.model_dirs.items()
    }
    complete = bool(ground_truth) and all(item["status"] == "ok" for item in models.values())
    correct_models = sum(bool(item["correct"]) for item in models.values())

    if not complete:
        status: TestStatus = "incomplete"
    elif correct_models >= consensus:
        status = "restored" if restored else "positive"
    else:
        status = "negative"

    return {
        "status": status,
        "correct_models": correct_models,
        "models": models,
    }


def write_id_file(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in ids:
            f.write(f"{item}\n")


def aggregate(args: argparse.Namespace) -> None:
    annotations = load_annotations(args.anno)
    test_configs = build_test_configs(
        Path(args.visual_output_root),
        Path(args.temporal_output_root),
        args.top_k,
        args.aggregation,
    )
    restore_path = Path(args.sensitivity_restore) if args.sensitivity_restore else None
    restore_ids = load_restore_ids(restore_path)
    known_ids = {sample_id(ann) for ann in annotations}
    unknown_restore_ids = sorted(restore_ids - known_ids)

    audio_contexts = ContextStore(
        "audio",
        transcript_dir=Path(args.transcript_dir),
    )

    diagnostic_results = []
    incomplete_results = []
    shortcut_ids = []
    test_counts: dict[str, Counter[str]] = defaultdict(Counter)
    model_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    unique_shortcut_counts: Counter[str] = Counter()

    for ann in annotations:
        sid = sample_id(ann)
        tests: dict[str, Any] = {}
        positive_tests = []

        for config in test_configs:
            applicable = not config.requires_audio or audio_contexts.get(ann) is not None
            result = evaluate_test(
                ann,
                config,
                consensus=args.consensus,
                applicable=applicable,
                restored=config.name == "frame_shuffle" and sid in restore_ids,
            )
            tests[config.name] = result
            test_counts[config.name][result["status"]] += 1
            if result["status"] == "positive":
                positive_tests.append(config.name)

            for model, model_result in result["models"].items():
                model_counts[config.name][model][model_result["status"]] += 1
                if model_result["status"] == "ok":
                    model_counts[config.name][model][
                        "correct" if model_result["correct"] else "incorrect"
                    ] += 1
                elif result["status"] != "not_applicable":
                    incomplete_results.append(
                        {
                            "sample_id": sid,
                            "test": config.name,
                            "model": model,
                            **model_result,
                        }
                    )

        is_shortcut = bool(positive_tests)
        if is_shortcut:
            shortcut_ids.append(sid)
        if len(positive_tests) == 1:
            unique_shortcut_counts[positive_tests[0]] += 1

        diagnostic_results.append(
            {
                "sample_id": sid,
                "db": ann.get("db"),
                "qid": ann.get("qid"),
                "ground_truth": ann.get("answer"),
                "tests": tests,
                "shortcut_tests": positive_tests,
                "is_shortcut": is_shortcut,
            }
        )

    summary_tests = {}
    for config in test_configs:
        per_model = {}
        for model in config.model_dirs:
            counts = model_counts[config.name][model]
            evaluated = counts["correct"] + counts["incorrect"]
            per_model[model] = {
                **dict(counts),
                "evaluated": evaluated,
                "accuracy": counts["correct"] / evaluated if evaluated else None,
            }
        summary_tests[config.name] = {
            **dict(test_counts[config.name]),
            "unique_shortcuts": unique_shortcut_counts[config.name],
            "models": per_model,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(diagnostic_results, output_dir / "diagnostic_results.json")
    dump_json(incomplete_results, output_dir / "incomplete_results.json")
    write_id_file(output_dir / "shortcut_ids.txt", shortcut_ids)
    dump_json(
        {
            "consensus": args.consensus,
            "top_k": args.top_k,
            "aggregation": args.aggregation,
            "annotations": len(annotations),
            "shortcuts": len(shortcut_ids),
            "remaining": len(annotations) - len(shortcut_ids),
            "sensitivity_restore_ids": len(restore_ids),
            "unknown_sensitivity_restore_ids": unknown_restore_ids,
            "incomplete_predictions": len(incomplete_results),
            "tests": summary_tests,
        },
        output_dir / "summary.json",
    )
    print(
        f"annotations={len(annotations)} shortcuts={len(shortcut_ids)} "
        f"remaining={len(annotations) - len(shortcut_ids)} "
        f"incomplete_predictions={len(incomplete_results)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate diagnostic model predictions into shortcut decisions."
    )
    parser.add_argument("--anno", default=str(DEFAULT_ANNO_PATH))
    parser.add_argument("--visual-output-root", default=str(DEFAULT_VISUAL_OUTPUT_ROOT))
    parser.add_argument("--temporal-output-root", default=str(DEFAULT_TEMPORAL_OUTPUT_ROOT))
    parser.add_argument("--transcript-dir", default=str(DATA_ROOT / "audios" / "stt"))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--consensus", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--sensitivity-restore")
    return parser.parse_args()


def main() -> None:
    aggregate(parse_args())


if __name__ == "__main__":
    main()
