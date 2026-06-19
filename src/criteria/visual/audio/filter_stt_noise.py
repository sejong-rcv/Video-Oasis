"""
STT Noise Filtering Script

This script is designed to filter out noisy or invalid files from STT (Speech-to-Text) data.
It separates clean transcripts from noisy ones by evaluating minimum length, hallucination keywords, valid character ratios, and unnatural word repetitions.
"""

import os
import json
import re
import tqdm
import shutil

SRC_ROOT = 'data/benchmarks/audios/stt' # raw STT files

DST_ROOT = 'data/benchmarks/audios/stt_clean' # directory for noise filtered STT

FILTERED_ROOT = 'data/benchmarks/audios/stt_filtered' # directory for filtered STT (option)

def is_valid_transcript(stt_data):
    """
    STT data validation (True: Valid / False: Noisy)
    """
    if not stt_data or "transcript" not in stt_data:
        return False

    full_text = stt_data.get("transcript", "").strip()

    # 1. Length check: Discard transcripts that are extremely short
    if len(full_text) < 4:
        return False

    # 2. Hallucination keyword detection: Filter out common STT artifacts
    HALLUCINATION_KEYWORDS = [
        "YTN", "Copyright",
        "Subtitles by", "Amara.org",
        "[Music]", "(Music)",
        "(Applause)", "[Applause]"
    ]
    for keyword in HALLUCINATION_KEYWORDS:
        if keyword.lower() in full_text.lower():
            return False

    # 3. Valid character ratio check: Filter out non-English or abnormal symbols
    allowed_pattern = re.compile(r"[a-zA-Z0-9\s\.,\?!'\"\(\)\[\]-]")
    allowed_char_count = len(allowed_pattern.findall(full_text))
    total_char_count = len(full_text)

    if total_char_count > 0:
        valid_ratio = allowed_char_count / total_char_count
        # Classify as noise if the ratio of valid characters is below 50% (e.g., emojis, corrupted text)
        if valid_ratio < 0.5:
            return False

    # 4. Repetition pattern check: Detect repetitive text generation (looping)
    words = full_text.split()
    if len(words) > 5:
        unique_words = set(words)
        ratio = len(unique_words) / len(words)
        # Discard if the unique word ratio is below 20%
        if ratio < 0.2:
            return False

    return True

def main():
    if not os.path.exists(SRC_ROOT):
        print(f"Error: Source directory '{SRC_ROOT}' not found.")
        return

    dataset_dirs = [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))]

    print(f"Found Datasets: {dataset_dirs}")
    print(f"Preprocessing STT files...")
    print(f" - Valid   -> '{DST_ROOT}'")
    print(f" - Filtered -> '{FILTERED_ROOT}'\n")

    total_files = 0
    kept_files = 0
    filtered_files = 0

    for dataset in dataset_dirs:
        src_dataset_path = os.path.join(SRC_ROOT, dataset)
        dst_dataset_path = os.path.join(DST_ROOT, dataset)
        filtered_dataset_path = os.path.join(FILTERED_ROOT, dataset)

        os.makedirs(dst_dataset_path, exist_ok=True)
        os.makedirs(filtered_dataset_path, exist_ok=True)

        files = [f for f in os.listdir(src_dataset_path) if f.endswith(".json")]

        print(f"   Processing '{dataset}' ({len(files)} files)...")

        for filename in tqdm.tqdm(files, desc=f"   Filtering {dataset}", leave=False):
            total_files += 1
            src_file = os.path.join(src_dataset_path, filename)
            dst_file = os.path.join(dst_dataset_path, filename)
            filtered_file = os.path.join(filtered_dataset_path, filename)

            try:
                with open(src_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if is_valid_transcript(data):
                    with open(dst_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)

                    if os.path.exists(filtered_file):
                        os.remove(filtered_file)

                    kept_files += 1
                else:
                    with open(filtered_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)

                    if os.path.exists(dst_file):
                        os.remove(dst_file)

                    filtered_files += 1

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print("\n" + "="*50)
    print("Preprocessing Completed!")
    print(f"Total Processed  : {total_files}")
    print(f"Kept (Clean)     : {kept_files}")
    print(f"Filtered (Noise) : {filtered_files}")
    print("-" * 50)
    print(f"Clean Data Path  : {DST_ROOT}")
    print(f"Filtered Data Path : {FILTERED_ROOT}")
    print("="*50)

if __name__ == "__main__":
    main()