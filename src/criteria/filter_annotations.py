"""Build a filtered annotation file by excluding shortcut sample ids."""

from __future__ import annotations

import argparse
from pathlib import Path

from criteria.common.annotations import dump_json, load_annotations, sample_id
from criteria.common.constants import DEFAULT_ANNO_PATH


def load_id_file(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                ids.add(item)
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter annotations by removing shortcut ids.")
    parser.add_argument("--anno", default=str(DEFAULT_ANNO_PATH))
    parser.add_argument("--shortcut-ids", required=True)
    parser.add_argument(
        "--exclude-ids",
        action="append",
        default=[],
        help="Additional reviewed id file to exclude; may be repeated.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.anno)
    shortcut_ids = load_id_file(args.shortcut_ids)
    additional_ids = set().union(*(load_id_file(path) for path in args.exclude_ids))
    excluded_ids = shortcut_ids | additional_ids
    filtered = [ann for ann in annotations if sample_id(ann) not in excluded_ids]
    dump_json(filtered, args.output)
    print(
        f"original={len(annotations)} shortcuts={len(shortcut_ids)} "
        f"additional_exclusions={len(additional_ids)} filtered={len(filtered)}"
    )


if __name__ == "__main__":
    main()
