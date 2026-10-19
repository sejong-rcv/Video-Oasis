"""Extract full-video audio and transcribe it with Whisper."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tqdm import tqdm

from criteria.common.annotations import dump_json, load_annotations, resolve_video_path
from criteria.common.constants import DATA_ROOT, DEFAULT_ANNO_PATH, REPO_ROOT
from criteria.visual.contexts import audio_path, load_transcript


def extract_audio(video: Path, output: Path) -> bool:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "json", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not json.loads(probe.stdout)["streams"]:
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp3")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                "-codec:a", "libmp3lame", "-q:a", "4", "-threads", "1",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if temporary.stat().st_size == 0:
            raise ValueError(f"Empty audio: {video}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def load_whisper(model: str, device: str):
    import torch
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        model_kwargs={"attn_implementation": "sdpa"},
    )


def run_stt(args: argparse.Namespace) -> dict[str, int]:
    annotations = load_annotations(args.anno)
    videos = {}
    for ann in annotations:
        path = audio_path(Path(args.output_dir), ann).with_suffix(".mp3.json")
        previous = videos.setdefault(path, ann)
        if previous["video_path"] != ann["video_path"]:
            raise ValueError(f"Transcript path collision: {path}")

    items = list(videos.items())
    if args.limit is not None:
        items = items[:args.limit]
    counts = dict(total=len(items), completed=0, skipped=0, no_audio=0, empty=0, failed=0)
    pipe = None

    for output, ann in tqdm(items, desc="stt"):
        if not args.overwrite and load_transcript(output, ann) is not None:
            counts["skipped"] += 1
            continue

        video = resolve_video_path(str(ann["video_path"]))
        mp3 = audio_path(Path(args.audio_dir), ann)
        try:
            if not extract_audio(video, mp3):
                counts["no_audio"] += 1
                tqdm.write(f"no_audio: {video}")
                continue
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            counts["failed"] += 1
            tqdm.write(f"audio_error: {video}: {getattr(exc, 'stderr', None) or exc}")
            continue

        if pipe is None:
            pipe = load_whisper(args.model, args.device)

        temporary = output.with_suffix(".json.tmp")
        try:
            result = pipe(
                str(mp3), chunk_length_s=30,
                batch_size=args.batch_size, return_timestamps=True,
            )
            text = result["text"]
            if not isinstance(text, str) or text.lstrip().startswith("Error:"):
                raise ValueError("Whisper returned an invalid transcript.")
            dump_json(
                {
                    "video_id": mp3.name,
                    "video_path": ann["video_path"],
                    "transcript": text,
                    "timestamps": result.get("chunks", []),
                },
                temporary,
            )
            temporary.replace(output)
            counts["completed"] += 1
            if not text.strip():
                counts["empty"] += 1
        except Exception as exc:
            counts["failed"] += 1
            tqdm.write(f"stt_error: {video}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)

    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", default=str(DEFAULT_ANNO_PATH))
    parser.add_argument("--model", default=str(REPO_ROOT / "data/models/whisper-large-v3"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audio-dir", default=str(DATA_ROOT / "audios/mp3"))
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "audios/stt"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(1 if run_stt(parse_args())["failed"] else 0)
