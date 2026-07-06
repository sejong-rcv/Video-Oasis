import os
import multiprocessing as mp
from tqdm import tqdm
from modules.data_loader import load_video_data
from modules.audio_utils import AudioProcessor

# cd Video-Oasis/src/preprocess/audio

# dataset json path (hardcoded)
JSON_PATH = "/home/jylee/workspace/Whisper/datasets/videomme_all.json" 

def main():
    try:
        data = load_video_data(JSON_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    unique_videos = {
      (item["db"].strip().lower(), item["video_path"])
      for item in data
  }

    audio_tasks = [
        (video_path, dataset)
        for dataset, video_path in unique_videos
    ]

    total_videos = len(audio_tasks)

    processor = AudioProcessor()

    print(f"\n[Phase 1] Audio Extraction Started")
    print(f" - Total Unique Videos: {total_videos}")
    print(f" - Output Directory: {processor.temp_dir}")
    print(f" - Workers: 16\n")

    with mp.Pool(16) as pool:
      results = list(
          tqdm(
              pool.imap(processor.extract_audio, audio_tasks),
              total=total_videos,
              desc="Extracting Audio",
          )
      )

    success_cnt = sum(1 for r in results if r is not None)
    fail_cnt = total_videos - success_cnt
    success_rate = (success_cnt / total_videos * 100) if total_videos > 0 else 0

    print("\n" + "="*40)
    print(f"  Extraction Summary")
    print("="*40)
    print(f"  - Total Videos   : {total_videos}")
    print(f"  - Success        : {success_cnt}")
    print(f"  - Failed         : {fail_cnt}")
    print(f"  - Success Rate   : {success_rate:.2f}%")
    print("="*40)

    if fail_cnt > 0:
        print(f" * Check your video files if 'Failed' count is high.")
    print(f"[Done] Audio extraction process completed.\n")

if __name__ == "__main__":
    main()