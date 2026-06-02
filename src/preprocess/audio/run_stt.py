import os
import json
import argparse
import math
import time
# [핵심] torch의 multiprocessing을 쓰면 CUDA 컨텍스트 관리가 더 안정적입니다.
import torch.multiprocessing as mp 
from tqdm import tqdm

MP3_DIR = "/home/jylee/workspace/Whisper/extracted_mp3"
STT_DIR = "/home/jylee/workspace/Whisper/STT"

def stt_worker(rank, gpu_id, file_chunk, batch_size, model_id):
    """
    rank: 프로세스 순번 (0, 1, 2...)
    gpu_id: 실제 할당할 GPU ID (0, 1, 2...)
    """
    # ---------------------------------------------------------
    # [중요] 프로세스 시작 직후에 라이브러리 임포트 (Lazy Import)
    # ---------------------------------------------------------
    try:
        import torch
        from modules.stt_utils import WhisperTranscriber
    except ImportError as e:
        print(f"[GPU-{gpu_id}] Import Error: {e}")
        return

    print(f"[GPU-{gpu_id}] (PID: {os.getpid()}) Initializing on Device {gpu_id}...")
    
    try:
        # device=int(gpu_id)를 직접 넘김
        transcriber = WhisperTranscriber(model_id=model_id, device=int(gpu_id))
    except Exception as e:
        print(f"[GPU-{gpu_id}] Model Load Failed: {e}")
        return

    print(f"[GPU-{gpu_id}] Start processing {len(file_chunk)} files.")
    
    success_count = 0
    for f in file_chunk:
        video_id = os.path.splitext(f)[0]
        save_path = os.path.join(STT_DIR, f"{video_id}.json")
        
        if os.path.exists(save_path):
            continue

        audio_path = os.path.join(MP3_DIR, f)
        try:
            result = transcriber.transcribe(audio_path, batch_size=batch_size)
            if not str(result.get("text", "")).startswith("Error:"):
                with open(save_path, 'w', encoding='utf-8') as out_f:
                    json.dump({
                        "video_id": video_id, 
                        "transcript": result["text"],
                        "timestamps": result.get("chunks", [])
                    }, out_f, indent=4, ensure_ascii=False)
                success_count += 1
        except Exception as e:
            print(f"[GPU-{gpu_id}] Error on {f}: {e}")

    print(f"[GPU-{gpu_id}] Finished! ({success_count} new files processed)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--batch_size", type=int, default=16) 
    args = parser.parse_args()

    gpu_list = [g.strip() for g in args.gpus.split(",")]
    num_gpus = len(gpu_list)

    # 1. 파일 목록 준비
    mp3_files = [f for f in os.listdir(MP3_DIR) if f.endswith(".mp3")]
    total_files = len(mp3_files)
    
    if total_files == 0:
        print("No files found.")
        return

    # 2. 일감 나누기 (Chunking)
    chunk_size = math.ceil(total_files / num_gpus)
    chunks = [mp3_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    print(f"\n[Parallel STT] Manual Process Mode | GPUs: {gpu_list}")
    
    # 3. [핵심] mp.Process 리스트 생성 및 시작
    # spawn 방식을 강제합니다.
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for i in range(num_gpus):
        # 만약 파일이 GPU 개수보다 적으면 에러 방지
        if i >= len(chunks): break
        
        p = mp.Process(
            target=stt_worker,
            args=(i, gpu_list[i], chunks[i], args.batch_size, args.model)
        )
        p.start()
        processes.append(p)
        print(f" -> Process {i} started for GPU {gpu_list[i]}")

    # 4. 모니터링 (tqdm)
    # Pool이 없으므로, 메인 프로세스는 파일 생성 개수만 감시합니다.
    with tqdm(total=total_files, desc="Total Progress") as pbar:
        prev = 0
        while True:
            # 살아있는 프로세스가 있는지 확인
            any_alive = any(p.is_alive() for p in processes)
            
            # 진행률 업데이트
            current = len([n for n in os.listdir(STT_DIR) if n.endswith(".json")])
            if current > prev:
                pbar.update(current - prev)
                prev = current
            
            if not any_alive and current >= prev:
                # 모든 프로세스가 죽었고 더 이상 파일도 안 늘어나면 종료
                break
            
            time.sleep(1)

    # 5. 뒷정리 (Join)
    for p in processes:
        p.join()
    
    print("\n[Done] All processes finished.")

if __name__ == "__main__":
    main()