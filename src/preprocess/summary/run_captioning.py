import torch
import os
import cv2
import json
import time

import numpy as np
import pickle
import tqdm
import argparse

from decord import VideoReader
from decord import cpu
# AutoCaptioner 임포트 경로는 선생님 환경에 맞게 유지
from care.models.modeling_captioners import AutoCaptioner

# ==========================================
# 1. Argument Parser 설정
# ==========================================
parser = argparse.ArgumentParser(description="Multi-GPU Captioning")
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (0-7)')
parser.add_argument('--total_gpus', type=int, default=8, help='Total number of GPUs/Chunks')
args = parser.parse_args()

print(f"🚀 Starting process on GPU {args.gpu_id} (Chunk {args.gpu_id + 1}/{args.total_gpus})")

if __name__ == '__main__':
    # 설정
    num_chunks_per_video = 8  # 비디오 하나를 8개 구간으로 쪼개서 캡션 생성
    frames_per_chunk = 1
    
    # 결과 저장 경로 설정 (GPU ID 별로 따로 저장)
    output_filename = f"./caption/total_summary_chunked_gpu{args.gpu_id}.json"
    
    # 이미 처리된 파일이 있으면 로드 (이어하기 기능)
    if os.path.isfile(output_filename):
        print(f"🔄 Found existing file {output_filename}, loading...")
        with open(output_filename, 'r') as f:
            total_dict = json.load(f)
    else:
        total_dict = dict()

    # 데이터 로드
    anno_path = "/mnt/users/gtlim/workspace/Video-Oasis/src/lmms_eval/video_total.json"
    total_anno = json.load(open(anno_path))
    
    # 모델 로드 (해당 GPU에 할당)
    device = f"cuda:{args.gpu_id}"
    captioner = AutoCaptioner.from_pretrained('/mnt/gtlim_data/users/gtlim/models/CaRe-7B', device_map=device)

    # 전체 비디오 리스트 생성
    video_list = set()
    for anno in total_anno:
        # 데이터셋 키 확인 ('video_path' or 'video')
        v_path = anno.get('video_path', anno.get('video'))
        if v_path:
            video_list.add(v_path)

    video_list = list(video_list)
    video_list.sort() # 정렬 필수! (모든 GPU가 같은 순서의 리스트를 봐야 함)

    # ==========================================
    # 2. Data Partitioning (전체 리스트 8등분)
    # ==========================================
    total_videos = len(video_list)
    chunk_size = total_videos // args.total_gpus
    remainder = total_videos % args.total_gpus
    
    start_idx = args.gpu_id * chunk_size + min(args.gpu_id, remainder)
    end_idx = start_idx + chunk_size + (1 if args.gpu_id < remainder else 0)
    
    # 내 GPU가 담당할 비디오 리스트
    my_video_list = video_list[start_idx:end_idx]
    
    print(f"📊 Total Videos: {total_videos}")
    print(f"📌 My Range: {start_idx} ~ {end_idx} (Count: {len(my_video_list)})")

    # ==========================================
    # 3. Processing
    # ==========================================
    # 주기적 저장을 위한 카운터
    save_interval = 10 
    
    for idx, video_path in enumerate(tqdm.tqdm(my_video_list, desc=f"GPU {args.gpu_id}")):
        
        # 이미 처리한 비디오면 스킵 (이어하기)
        if video_path in total_dict:
            continue

        total_dict[video_path] = dict()
        try:
            total_dict[video_path]['db'] = video_path.split('/')[-3]
            total_dict[video_path]['vid'] = video_path.split('/')[-1]
        except:
            total_dict[video_path]['db'] = 'unknown'
            total_dict[video_path]['vid'] = os.path.basename(video_path)

        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=4) # CPU 쓰레드 과부하 방지 위해 줄임
            total_frames = len(vr)
            
            chunk_inputs = []

            for i in range(num_chunks_per_video):
                start_frame = (total_frames * i) // num_chunks_per_video
                end_frame = (total_frames * (i + 1)) // num_chunks_per_video

                # 구간 내 16프레임 샘플링
                if frames_per_chunk > 1:
                    frame_indices = np.linspace(start_frame, end_frame - 1, frames_per_chunk, dtype=int)
                    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
                else:
                    frame_indices = [int((start_frame + end_frame)/2)]

                frames = vr.get_batch(frame_indices).asnumpy()
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2) # (16, C, H, W)
                chunk_inputs.append(frames_tensor)

            # (N, T, C, H, W) 형태로 배치 생성
            batch_input = torch.stack(chunk_inputs).to(device)

            with torch.no_grad():
                # 모델 추론
                descriptions = captioner.describe(batch_input)
            
            total_dict[video_path]['summary'] = descriptions

        except Exception as e:
            print(f"⚠️ Error on {video_path}: {e}")
            total_dict[video_path]['summary'] = ['NONE'] * num_chunks_per_video
        
        if (idx + 1) % save_interval == 0:
             with open(output_filename, "w") as json_file:
                json.dump(total_dict, json_file, indent=4)

    # 최종 저장
    with open(output_filename, "w") as json_file:
        json.dump(total_dict, json_file, indent=4)
    
    print(f"✅ GPU {args.gpu_id} Finished! Saved to {output_filename}")