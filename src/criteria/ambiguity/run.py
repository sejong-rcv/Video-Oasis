"""Command-line entry point for annotation-ambiguity diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from criteria.ambiguity.consistency import parse_prediction_source, run_consistency
from criteria.ambiguity.redundancy import RedundancyConfig, run_redundancy
from criteria.ambiguity.sensitivity import build_sensitivity_queue
from criteria.common.annotations import load_annotations
from criteria.common.constants import DEFAULT_ANNO_PATH


AMBIGUITY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = AMBIGUITY_ROOT / "output"
DEFAULT_DIAGNOSTIC_REPORT = AMBIGUITY_ROOT.parent / "reports" / "diagnostic_results.json"
DEFAULT_OASIS_ANNO = DEFAULT_ANNO_PATH.with_name("video_oasis.json")


def run_consistency_command(args: argparse.Namespace) -> None:
    run_consistency(
        annotations=load_annotations(args.anno),
        sources=[parse_prediction_source(value) for value in args.prediction],
        output_dir=Path(args.output_dir),
        rule=args.rule,
    )


def run_sensitivity_command(args: argparse.Namespace) -> None:
    build_sensitivity_queue(
        annotations=load_annotations(args.anno),
        diagnostic_report=Path(args.diagnostic_report),
        output_dir=Path(args.output_dir),
    )


def run_redundancy_command(args: argparse.Namespace) -> None:
    run_redundancy(
        RedundancyConfig(
            anno_path=Path(args.anno),
            output_dir=Path(args.output_dir),
            model=args.model,
            device=args.device,
            num_chunks=args.num_chunks,
            frames_per_chunk=args.frames_per_chunk,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
            summarize_only=args.summarize_only,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Video-Oasis ambiguity diagnostics.")
    tests = parser.add_subparsers(dest="test", required=True)

    consistency = tests.add_parser("consistency", help="Analyze five-model disagreement.")
    consistency.add_argument("--anno", default=str(DEFAULT_OASIS_ANNO))
    consistency.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="MODEL=PREDICTION_PATH",
    )
    consistency.add_argument(
        "--rule",
        choices=(
            "maximal-disagreement",
            "option-coverage",
            "no-majority",
            "all-unique",
        ),
        default="maximal-disagreement",
    )
    consistency.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "consistency"))
    consistency.set_defaults(handler=run_consistency_command)

    redundancy = tests.add_parser("redundancy", help="Run Eagle2.5 on eight video chunks.")
    redundancy.add_argument("--anno", default=str(DEFAULT_OASIS_ANNO))
    redundancy.add_argument("--model", choices=("eagle25",), default="eagle25")
    redundancy.add_argument("--device", default="cuda")
    redundancy.add_argument("--num-chunks", type=int, default=8)
    redundancy.add_argument("--frames-per-chunk", type=int, default=16)
    redundancy.add_argument("--max-new-tokens", type=int, default=1024)
    redundancy.add_argument("--overwrite", action="store_true")
    redundancy.add_argument("--summarize-only", action="store_true")
    redundancy.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "redundancy"))
    redundancy.set_defaults(handler=run_redundancy_command)

    sensitivity = tests.add_parser("sensitivity", help="Build a Frame Shuffling review queue.")
    sensitivity.add_argument("--anno", default=str(DEFAULT_ANNO_PATH))
    sensitivity.add_argument("--diagnostic-report", default=str(DEFAULT_DIAGNOSTIC_REPORT))
    sensitivity.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "sensitivity"))
    sensitivity.set_defaults(handler=run_sensitivity_command)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
