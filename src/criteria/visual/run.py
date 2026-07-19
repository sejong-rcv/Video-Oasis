"""Command-line entry point for visual-dependency tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from criteria.common.constants import DATA_ROOT, DEFAULT_ANNO_PATH
from criteria.common.text import text_model_names
from criteria.visual.runner_text import VisualTextConfig, run_visual_test


DEFAULT_SUMMARY_FILE = Path(__file__).with_name("total_summary.json")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--anno", default=str(DEFAULT_ANNO_PATH))
    parser.add_argument("--model", required=True, choices=text_model_names())
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
