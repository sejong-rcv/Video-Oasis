"""Shared constants used by the diagnostic criteria draft."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "benchmarks"
DEFAULT_ANNO_PATH = REPO_ROOT / "src" / "lmms_eval" / "video_total.json"
DEFAULT_FEATURE_DIR = DATA_ROOT / "features"

ANSWER_LETTERS = tuple("ABCDEFGHIJKLMN")

VISUAL_TESTS = ("blind", "audio", "summary")
TEMPORAL_TESTS = ("center_frame", "frame_shuffle", "bag_of_frames")
AMBIGUITY_TESTS = ("consistency", "redundancy", "sensitivity")
