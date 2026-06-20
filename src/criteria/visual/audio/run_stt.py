import os
import json
import argparse
import math
import time
import torch.multiprocessing as mp
from tqdm import tqdm

MP3_ROOT = "data/benchmarks/audios/mp3"
STT_ROOT = "data/benchmarks/audios/stt"

def stt_worker(rank, gpu_id, file_chunk, batch_size, model_id):

    try:
        import torch
        from modules.stt_utils import WhisperTranscriber
    except ImportError as e:
        print(f"[GPU-{gpu_id}] Import Error: {e}")
        return

    print(f"[GPU-{gpu_id}] (PID: {os.getpid()}) Initializing on Device {gpu_id}...")

    try:
        transcriber = WhisperTranscriber(model_id=model_id, device=int(gpu_id))
    except Exception as e:
        print(f"[GPU-{gpu_id}] Model Load Failed: {e}")
        return

    print(f"[GPU-{gpu_id}] Start processing {len(file_chunk)} files.")

    success_count = 0
    for dataset, filename in file_chunk:
        video_id = os.path.splitext(filename)[0]
        audio_path = os.path.join(MP3_ROOT, dataset, filename)
        dataset_stt_dir = os.path.join(STT_ROOT, dataset)
        save_path = os.path.join(dataset_stt_dir, f"{filename}.json")

        if os.path.exists(save_path):
            continue

        try:
            result = transcriber.transcribe(audio_path, batch_size=batch_size)
            if not str(result.get("text", "")).startswith("Error:"):
                with open(save_path, 'w', encoding='utf-8') as out_f:
                    json.dump({
                        "video_id": filename,
                        "transcript": result["text"],
                        "timestamps": result.get("chunks", [])
                    }, out_f, indent=4, ensure_ascii=False)
                success_count += 1
        except Exception as e:
            print(f"[GPU-{gpu_id}] Error on {audio_path}: {e}")

    print(f"[GPU-{gpu_id}] Finished! ({success_count} new files processed)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    gpu_list = [g.strip() for g in args.gpus.split(",")]
    num_gpus = len(gpu_list)

    if not os.path.isdir(MP3_ROOT):
        print(f"MP3 root directory not found: {MP3_ROOT}")
        return

    dataset_dirs = sorted(
        dataset
        for dataset in os.listdir(MP3_ROOT)
        if os.path.isdir(os.path.join(MP3_ROOT, dataset))
    )

    audio_tasks = []
    for dataset in dataset_dirs:
        dataset_mp3_dir = os.path.join(MP3_ROOT, dataset)
        dataset_stt_dir = os.path.join(STT_ROOT, dataset)
        os.makedirs(dataset_stt_dir, exist_ok=True)

        filenames = sorted(
            filename
            for filename in os.listdir(dataset_mp3_dir)
            if filename.lower().endswith(".mp3")
        )
        audio_tasks.extend((dataset, filename) for filename in filenames)

    total_files = len(audio_tasks)

    if total_files == 0:
        print(f"No MP3 files found under: {MP3_ROOT}")
        return

    chunk_size = math.ceil(total_files / num_gpus)
    chunks = [
        audio_tasks[i:i + chunk_size]
        for i in range(0, total_files, chunk_size)
    ]

    print(f"\n[Parallel STT] Manual Process Mode | GPUs: {gpu_list}")
    print(f" - Datasets: {dataset_dirs}")
    print(f" - Total MP3 Files: {total_files}")
    print(f" - STT Output Root: {STT_ROOT}")

    mp.set_start_method('spawn', force=True)

    processes = []
    for i in range(num_gpus):
        if i >= len(chunks): break

        p = mp.Process(
            target=stt_worker,
            args=(i, gpu_list[i], chunks[i], args.batch_size, args.model)
        )
        p.start()
        processes.append(p)
        print(f" -> Process {i} started for GPU {gpu_list[i]}")

    with tqdm(total=total_files, desc="Total Progress") as pbar:
        def count_completed():
            return sum(
                os.path.exists(
                    os.path.join(
                        STT_ROOT,
                        dataset,
                        f"{filename}.json",
                    )
                )
                for dataset, filename in audio_tasks
            )

        prev = count_completed()
        pbar.update(prev)

        while True:
            any_alive = any(p.is_alive() for p in processes)

            current = count_completed()
            if current > prev:
                pbar.update(current - prev)
                prev = current

            if not any_alive and current >= prev:
                break

            time.sleep(1)

    for p in processes:
        p.join()

    print("\n[Done] All processes finished.")

if __name__ == "__main__":
    main()