"""Draft CLI for ambiguity checks."""

from __future__ import annotations

import argparse

from criteria.ambiguity.specs import AMBIGUITY_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ambiguity diagnostic check.")
    parser.add_argument("--test", required=True, choices=sorted(AMBIGUITY_SPECS))
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = AMBIGUITY_SPECS[args.test]
    raise NotImplementedError(
        f"{spec.paper_name} is not migrated yet. "
        "Ambiguity checks should consume prediction files rather than load models directly."
    )


if __name__ == "__main__":
    main()
