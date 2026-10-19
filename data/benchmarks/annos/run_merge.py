import argparse
import json
import math
import os
import re
from collections import Counter
from numbers import Real
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd


SOURCE_FILES = {
    "ImplicitQA": ("implicitqa/ImplicitQAv0.1.2.jsonl", "question_id"),
    "LVBench": ("lvbench/video_info.meta.jsonl", "uid"),
    "LongVideoBench": ("longvideobench/lvb_val.json", "id"),
    "MLVU_Test": ("mlvu_test/test_mcq_gt.json", "question_id"),
    "MVBench": ("mvbench/json/*.json", None),
    "RTV-Bench": ("rtv-bench/QA.json", "questionID"),
    "TVBench": ("tvbench/json/*.json", None),
    "VCR-Bench": ("vcr-bench/VCR-Bench.tsv", "id"),
    "VSI-Bench": ("vsi-bench/test.jsonl", "id"),
    "Video-Holmes": ("video-holmes/test_Video-Holmes.json", "Question ID"),
    "Video-MME": ("video-mme/test-00000-of-00001.parquet", "question_id"),
    "egoschema": ("egoschema/test-00000-of-00001.parquet", "question_idx"),
    "MINERVA": ("minerva/minerva.json", "key"),
    "MMR-V": ("mmrvbench/test-00000-of-00001.parquet", "question_idx"),
}

# These unavailable videos still belong to the released QA set.
KEEP_MISSING_VIDEOS = {
    ("ImplicitQA", "mD_uGyelMW4.mp4"),
    ("MINERVA", "03hFktDJ6zc.mp4"),
}


def read_rows(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t").to_dict("records")
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def normalize_options(options):
    normalized = []
    for index, option in enumerate(options):
        letter = chr(65 + index)
        if not option.strip():
            normalized.append(f"{letter}. ")
            continue
        match = re.fullmatch(r"\s*\([A-Z]\)\s*(.*)", option, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Invalid source option label: {option!r}")
        text = re.sub(r"^\(Correct\)\s*", "", match[1], flags=re.IGNORECASE)
        text = re.sub(r"^[A-Z]\)\s+", "", text).strip()
        normalized.append(f"{letter}. {text}")
    return normalized


def interval(sample, start_key, end_key):
    if start_key not in sample and end_key not in sample:
        return None
    if start_key not in sample or end_key not in sample:
        raise ValueError(f"Incomplete interval: {start_key}/{end_key}")
    values = sample[start_key], sample[end_key]
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError(f"Interval values must be numeric: {values}")
    start, end = map(float, values)
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError(f"Invalid interval: {values}")
    return start, end


def make_qa(db, sample):
    meta = ""
    bounds = None
    extra = {}
    if db == "ImplicitQA":
        question = sample["question_text"]
        video = parse_qs(urlsplit(sample["video_url"]).query)["v"][0]
        options = [f"{key}. {value}" for key, value in sample["options"].items()]
        answer = sample["answer_choice"]
        meta = f"\nCategory : {sample['category']}"
        bounds = interval(sample, "question_start_time", "question_stop_time")

    elif db == "LVBench":
        qa = sample["qa"]
        question, *raw_options = qa["question"].splitlines()
        video = sample["key"]
        options = normalize_options(raw_options)
        answer = qa["answer"]
        meta = (
            f"\nType : {'. '.join(qa['question_type'])} \nGenre : {sample['type']}"
            f"\nDuration (min) : {sample['video_info']['duration_minutes']}"
        )

    elif db == "LongVideoBench":
        question, video = sample["question"], sample["video_id"]
        options = [f"{chr(65 + i)}. {value}" for i, value in enumerate(sample["candidates"])]
        answer = chr(65 + sample["correct_choice"])
        meta = f"\nTopic : {sample['topic_category']}\nDuration (sec) : {sample['duration']}"

    elif db in {"MLVU_Test", "MVBench", "TVBench"}:
        question, video = sample["question"], sample["video"]
        options = [f"{chr(65 + i)}. {value}" for i, value in enumerate(sample["candidates"])]
        answer = None
        for i, value in enumerate(sample["candidates"]):
            if value == sample["answer"]:
                answer = chr(65 + i)
        if db == "MLVU_Test":
            meta = f"\nType : {sample['question_type']}\nDuration (sec) : {sample['duration']}"
        else:
            bounds = interval(sample, "start", "end")

    elif db == "RTV-Bench":
        question, video = sample["question"], sample["video"]
        options = [f"{key}. {value}" for key, value in sample["options"].items()]
        answer = sample["answer"]
        meta = f"\nType : {sample['type']}\nField : {sample['field']}"
        bounds = interval(sample, "start_time", "end_time")

    elif db == "VCR-Bench":
        question, *lines = sample["question"].splitlines()
        options = lines[:-1]
        video = sample["video_path"]
        answer = sample["answer"]
        meta = f"\nType : {sample['dimension']}\nReasoning : {sample['reasoning']}"

    elif db == "VSI-Bench":
        if sample["options"] is None:
            return None
        question = sample["question"]
        video = f"{sample['dataset']}_{sample['scene_name']}"
        options = list(sample["options"])
        answer = sample["ground_truth"]
        meta = f"\nCategory : {sample['question_type']}"

    elif db == "Video-Holmes":
        type_dict = {
            "SR": "Social Reasoning (Inferring social relationships between characters. This includes identifying identity associations across time (e.g., the same man in youth and old age))",
            "PAR": "Physical Anomaly Reasoning (Identifying scenes in the video that deviate from real-world norms and reasoning about their underlying rules or implicit meanings)",
            "MHR": "Multimodal Hint Reasoning (Decoding cues or fact from multimodal hints, such as semantic implications of camera movements or gradual changes in object positions)",
            "IMC": "Intention & Motive Chaining (Observing characters\u2019 actions or environmental cues to disentangle surface behaviors from underlying behavioral intentions)",
            "TCI": "Temporal Causal Inference (Inferring causal mechanisms between events across time and space using cinematic language and multimodal clues)",
            "TA": "Timeline Analysis (Integrating and reconstructing the narrative storyline of the film)",
            "CTI": "Core Theme Inference (Extracting the core theme or deeper meaning of the video by analyzing its plot, dialogues, and symbolic elements)",
        }
        question, video = sample["Question"], sample["video ID"]
        options = [f"{key}. {value}" for key, value in sample["Options"].items()]
        answer = sample["Answer"]
        meta = f"\nType of QA :{type_dict[sample['Question Type']]}\nReasoning : {sample['Explanation']}"

    elif db == "Video-MME":
        question, video = sample["question"], sample["videoID"]
        options = list(sample["options"])
        answer = sample["answer"]
        meta = f"\nDomain : {sample['domain']}\nDuration : {sample['duration']}\nCategory : {sample['sub_category']}"

    elif db == "egoschema":
        question, video = sample["question"], sample["video_idx"]
        options = list(sample["option"])
        answer = chr(int(sample["answer"]) + 65)

    elif db == "MINERVA":
        question, video = sample["question"], sample["video_id"]
        options = [f"{chr(65 + i)}. {sample[f'answer_choice_{i}']}" for i in range(5)]
        answer = chr(sample["answer_id"] + 65)
        meta = f"\nType of QA : {sample['question_type']}\nReasoning : {sample['reasoning']}"

    elif db == "MMR-V":
        question, video = sample["question"], sample["video"]
        options = normalize_options(sample["options"])
        answer = sample["correctAnswer"].strip("()")
        extra = {key: sample[key] for key in ("abilityType_L2", "abilityType_L3")}

    else:
        raise ValueError(f"Unknown benchmark: {db}")

    letters = [chr(65 + i) for i in range(len(options))]
    # Exclude free-form questions and invalid multiple-choice annotations.
    if len(options) < 2 or answer not in letters:
        return None
    if any(not option.startswith(f"{letter}. ") for letter, option in zip(letters, options)):
        raise ValueError("Option labels must be consecutive from A")
    answer_text = options[ord(answer) - 65][3:]
    if db in {"VCR-Bench", "Video-Holmes", "MINERVA"}:
        meta += (" \nAnswer : " if db == "VCR-Bench" else "\nAnswer : ") + answer_text
    separator = " " if db in {"TVBench", "VCR-Bench", "Video-MME", "egoschema", "MINERVA", "MMR-V"} else ". "
    meta = "Options of QA : " + "".join(option + separator for option in options) + meta

    if Path(video).suffix.lower() not in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
        video += ".mp4"
    folder = "mmrvbench" if db == "MMR-V" else db.lower()
    filename = video
    if db == "VCR-Bench":
        filename = video.removeprefix("v1/videos/video/")
    elif db == "VSI-Bench":
        filename = f"{sample['dataset']}/{sample['scene_name']}.mp4"
    elif db == "LongVideoBench" and video.startswith("@"):
        filename = video.rsplit("-", 1)[-1]

    row = {
        "question": question,
        "video": video,
        "options": options,
        "answer": answer,
        "answer_text": answer_text,
        "meta": meta,
        **extra,
        "db": db,
        "video_path": f"../data/benchmarks/videos/{folder}/videos/{filename}",
    }
    if db in {"ImplicitQA", "RTV-Bench"} and bounds is None:
        raise ValueError(f"Missing required source interval: {db}")
    if bounds is not None:
        row["start_time"], row["end_time"] = bounds
    return row


def get_source_files(source_dir):
    sources = []
    for db, (pattern, id_field) in SOURCE_FILES.items():
        paths = sorted(source_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"Missing source annotations: {source_dir / pattern}")
        sources.extend((db, path, id_field) for path in paths)
    return sources


def merge_annotations(sources, video_dir):
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing video directory: {video_dir}")
    rows = []
    seen = set()
    video_counts = Counter()
    for db, path, id_field in sources:
        data = read_rows(path)
        if db == "LVBench":
            data = [{**sample, "qa": qa} for sample in data for qa in sample["qa"]]
        for index, sample in enumerate(data):
            if db in {"MVBench", "TVBench"}:
                # Both benchmarks share the counter; assign IDs before filtering.
                video = sample["video"]
                qid = video + "@" * video_counts[video]
                video_counts[video] += 1
            else:
                qa = sample["qa"] if db == "LVBench" else sample
                qid = str(qa[id_field])
            if (db, qid) in seen:
                raise ValueError(f"Duplicate QA ID: {db}/{qid}")
            seen.add((db, qid))
            try:
                row = make_qa(db, sample)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source QA: {db}/{qid} ({path}:{index}): {exc}") from exc
            if row is None:
                continue
            video_path = video_dir / Path(row["video_path"]).relative_to("../data/benchmarks/videos")
            if not video_path.is_file() and (db, row["video"]) not in KEEP_MISSING_VIDEOS:
                continue
            row["qid"] = qid
            rows.append(row)
    return rows


def write_json(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=4, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    source_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Merge benchmark annotations into video_total.json.")
    parser.add_argument("--source-dir", type=Path, default=source_dir)
    parser.add_argument("--output", type=Path, default=source_dir / "video_total.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = get_source_files(args.source_dir)
    inputs = [Path(__file__), *(path for _, path, _ in sources)]
    if args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError(f"Refusing to overwrite an input file: {args.output}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite: {args.output}")

    rows = merge_annotations(sources, args.source_dir.parent / "videos")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, rows)
    for db, count in Counter(row["db"] for row in rows).items():
        print(f"DB : {db:25s} || Num of QA : {count:5d}")
    print(f"# of Total QA : {len(rows)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
