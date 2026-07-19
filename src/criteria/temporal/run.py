"""Command-line entry point for temporal-dependency tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from criteria.common.constants import DEFAULT_ANNO_PATH, DEFAULT_FEATURE_DIR
from criteria.common.models import model_names
from criteria.temporal.runner_vlm import (
    BagOfFramesConfig,
    run_bag_of_frames,
)
from criteria.temporal.runner_mllm import (
    TemporalMLLMConfig,
    run_mllm_test,
)


def add_common_arguments(parser: argparse.ArgumentParser, backend: str) -> None:
    parser.add_argument("--anno", default=str(DEFAULT_ANNO_PATH))
    parser.add_argument("--model", required=True, choices=model_names(backend))
    parser.add_argument("--device", default="cuda")


def add_mllm_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser, backend="mllm")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")


def run_center_frame(args: argparse.Namespace) -> None:
    run_mllm_test(
        TemporalMLLMConfig(
            test="center_frame",
            model=args.model,
            anno_path=Path(args.anno),
            output_dir=Path(args.output_dir),
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
    )


def run_frame_shuffle(args: argparse.Namespace) -> None:
    run_mllm_test(
        TemporalMLLMConfig(
            test="frame_shuffle",
            model=args.model,
            anno_path=Path(args.anno),
            output_dir=Path(args.output_dir),
            device=args.device,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
    )


def run_bof(args: argparse.Namespace) -> None:
    run_bag_of_frames(
        BagOfFramesConfig(
            model=args.model,
            anno_path=Path(args.anno),
            feature_dir=Path(args.feature_dir),
            output_dir=Path(args.output_dir),
            device=args.device,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            top_k=args.top_k,
            aggregation=args.aggregation,
            max_text_length=args.max_text_length,
            overwrite=args.overwrite,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Video-Oasis temporal-dependency diagnostic test."
    )
    tests = parser.add_subparsers(dest="test", required=True)

    center = tests.add_parser("center-frame", help="Run the Center-Frame test.")
    add_mllm_arguments(center)
    center.add_argument("--output-dir", default="output/center_frame")
    center.set_defaults(handler=run_center_frame)

    shuffle = tests.add_parser("frame-shuffle", help="Run the Frame Shuffling test.")
    add_mllm_arguments(shuffle)
    shuffle.add_argument("--num-frames", type=int, default=128)
    shuffle.add_argument("--output-dir", default="output/frame_shuffle")
    shuffle.set_defaults(handler=run_frame_shuffle)

    bof = tests.add_parser("bag-of-frames", help="Run the Bag-of-Frames test.")
    add_common_arguments(bof, backend="vlm")
    bof.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    bof.add_argument("--output-dir", default="output/bag_of_frames")
    bof.add_argument("--batch-size", type=int, default=32)
    bof.add_argument("--max-frames", type=int, default=2048)
    bof.add_argument("--top-k", type=int, default=32)
    bof.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    bof.add_argument("--max-text-length", type=int, default=64)
    bof.add_argument("--overwrite", action="store_true")
    bof.set_defaults(handler=run_bof)

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
