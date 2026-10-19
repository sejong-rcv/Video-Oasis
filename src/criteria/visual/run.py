"""Inference and command-line entry point for visual-dependency tests."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from criteria.common.annotations import load_annotations
from criteria.common.constants import DATA_ROOT, DEFAULT_ANNO_PATH
from criteria.common.mllm import generate_text, load_mllm
from criteria.common.models import model_names
from criteria.common.outputs import load_saved_prediction, prediction_path, save_prediction
from criteria.common.parsing import extract_answer_letter
from criteria.common.prompts import AUDIO_PROMPT_VERSION, build_audio_prompt, build_visual_dependency_prompt
from criteria.visual.contexts import ContextStore, VisualTest


DEFAULT_SUMMARY_FILE = Path(__file__).with_name("total_summary.json")


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


def parse_audio_response(response: str, num_options: int) -> tuple[str, str, str]:
    text = re.split(r"</think>", response, flags=re.IGNORECASE)[-1]
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        return "Unknown", "", ""
    parts = re.split(r"^[ \t]*(Answer|Evidence|Reasoning):[ \t]*", text,
                     flags=re.MULTILINE | re.IGNORECASE)
    fields = {}
    for name, value in zip(parts[1::2], parts[2::2]):
        name = name.lower()
        if name in fields:
            return "Unknown", "", ""
        fields[name] = value.strip()
    prediction = extract_answer_letter(fields.get("answer", ""), num_options)
    return prediction or "Unknown", fields.get("evidence", ""), fields.get("reasoning", "")


def run_visual_test(config: VisualTextConfig) -> VisualRunSummary:
    annotations = load_annotations(config.anno_path)
    contexts = ContextStore(
        test=config.test,
        transcript_dir=config.transcript_dir,
        summary_file=config.summary_file,
    )
    runtime = None

    completed = 0
    skipped = 0
    missing_context = 0
    failed = 0

    for ann in tqdm(annotations, desc=f"{config.test}:{config.model}"):
        output_path = prediction_path(config.output_dir, config.model, ann)
        saved = load_saved_prediction(output_path) if not config.overwrite else None
        if saved is not None and config.test == "audio":
            diagnostic = saved.get("diagnostic")
            if not isinstance(diagnostic, dict) or diagnostic.get("prompt_version") != AUDIO_PROMPT_VERSION:
                saved = None
        if saved is not None:
            skipped += 1
            continue

        if runtime is None:
            runtime = load_mllm(config.model, config.device)

        try:
            context = contexts.get(ann)
            if context is None:
                missing_context += 1
                continue
            if config.test == "audio":
                prompt = build_audio_prompt(ann, context.text)
            else:
                prompt = build_visual_dependency_prompt(
                    ann, context_label=context.label, context=context.text,
                )
            prediction = generate_text(
                runtime,
                prompt,
                max_new_tokens=config.max_new_tokens,
            )
            diagnostic = {
                "axis": "visual",
                "test": config.test,
                "model": config.model,
                "context_source": context.source,
            }
            payload = ann
            if config.test == "audio":
                raw_response = prediction
                prediction, evidence, reasoning = parse_audio_response(raw_response, len(ann["options"]))
                payload = {**ann, "raw_response": raw_response, "Evidence": evidence, "Reasoning": reasoning}
                diagnostic["prompt_version"] = AUDIO_PROMPT_VERSION
            save_prediction(output_path, payload, prediction, diagnostic=diagnostic)
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


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--anno", default=str(DEFAULT_ANNO_PATH))
    parser.add_argument("--model", required=True, choices=model_names("mllm"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")


def run_test(args: argparse.Namespace) -> None:
    run_visual_test(
        VisualTextConfig(
            test=args.test,
            model=args.model,
            anno_path=Path(args.anno),
            output_dir=Path(args.output_dir),
            device=args.device,
            transcript_dir=Path(args.transcript_dir) if args.transcript_dir else None,
            summary_file=Path(args.summary_file) if args.summary_file else None,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Video-Oasis visual-dependency diagnostic test."
    )
    tests = parser.add_subparsers(dest="test", required=True)

    blind = tests.add_parser("blind", help="Run the Blind Test.")
    add_common_arguments(blind)
    blind.add_argument("--output-dir", default="output/blind")
    blind.set_defaults(handler=run_test, transcript_dir=None, summary_file=None)

    audio = tests.add_parser("audio", help="Run the Audio Test.")
    add_common_arguments(audio)
    audio.add_argument("--transcript-dir", default=str(DATA_ROOT / "audios" / "stt"))
    audio.add_argument("--output-dir", default="output/audio")
    audio.set_defaults(handler=run_test, summary_file=None)

    summary = tests.add_parser("summary", help="Run the Summary Test.")
    add_common_arguments(summary)
    summary.add_argument("--summary-file", default=str(DEFAULT_SUMMARY_FILE))
    summary.add_argument("--output-dir", default="output/summary")
    summary.set_defaults(handler=run_test, transcript_dir=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
