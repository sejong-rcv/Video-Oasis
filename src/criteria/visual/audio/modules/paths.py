from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
AUDIO_ROOT = REPO_ROOT / "data" / "benchmarks" / "audios"

MP3_ROOT = AUDIO_ROOT / "mp3"
STT_ROOT = AUDIO_ROOT / "stt"
STT_CLEAN_ROOT = AUDIO_ROOT / "stt_clean"
STT_FILTERED_ROOT = AUDIO_ROOT / "stt_filtered"

AUDIO_CRITERIA_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = AUDIO_CRITERIA_ROOT / "output_prior"

