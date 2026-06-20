import torch
import os
import json
import numpy as np
import tqdm
import argparse

from decord import VideoReader
from decord import cpu
from care.models.modeling_captioners import AutoCaptioner

parser = argparse.ArgumentParser(description="Multi-GPU Captioning")
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (0-7)')
parser.add_argument('--total_gpus', type=int, default=8, help='Total number of GPUs/Chunks')
args = parser.parse_args()

if __name__ == '__main__':
    num_chunks_per_video = 8 
    frames_per_chunk = 16
    
    output_filename = f"/data3/gtlim/workspace/src/Video-Oasis/src/criteria/visual/summary/caption/total_summary_chunked_gpu{args.gpu_id}.json"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    if os.path.isfile(output_filename):
        print(f"Found existing file {output_filename}, loading...")
        with open(output_filename, 'r') as f:
            total_dict = json.load(f)
    else:
        total_dict = dict()

    anno_path = "/data3/gtlim/workspace/src/Video-Oasis/src/lmms_eval/video_total.json"
    total_anno = json.load(open(anno_path))
    
    device = f"cuda:{args.gpu_id}"
    captioner = AutoCaptioner.from_pretrained('/data3/gtlim/workspace/src/Video-Oasis/data/models/CaRe-7B', device_map=device)

    video_list = set()
    for anno in total_anno:
        original_path = anno.get('video_path', anno.get('video'))
        if original_path:
            summary_key = original_path.replace('../data/benchmarks', '')
            video_path = original_path.replace(
                '../data/benchmarks',
                '/data3/gtlim/workspace/src/Video-Oasis/data/benchmarks',
            )
            video_list.add((summary_key, video_path))

    video_list = list(video_list)
    video_list.sort()

    total_videos = len(video_list)
    chunk_size = total_videos // args.total_gpus
    remainder = total_videos % args.total_gpus
    
    start_idx = args.gpu_id * chunk_size + min(args.gpu_id, remainder)
    end_idx = start_idx + chunk_size + (1 if args.gpu_id < remainder else 0)
    
    my_video_list = video_list[start_idx:end_idx]
    
    print(f"Total Videos: {total_videos}")
    print(f"My Range: {start_idx} ~ {end_idx} (Count: {len(my_video_list)})")

    save_interval = 10 
    
    for idx, (summary_key, video_path) in enumerate(tqdm.tqdm(my_video_list, desc=f"GPU {args.gpu_id}")):
        
        if summary_key in total_dict:
            continue

        total_dict[summary_key] = dict()
        try:
            total_dict[summary_key]['db'] = summary_key.split('/')[-3]
            total_dict[summary_key]['vid'] = summary_key.split('/')[-1]
        except:
            total_dict[summary_key]['db'] = 'unknown'
            total_dict[summary_key]['vid'] = os.path.basename(summary_key)

        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
            total_frames = len(vr)
            
            chunk_inputs = []

            for i in range(num_chunks_per_video):
                start_frame = (total_frames * i) // num_chunks_per_video
                end_frame = (total_frames * (i + 1)) // num_chunks_per_video

                if frames_per_chunk > 1:
                    frame_indices = np.linspace(start_frame, end_frame - 1, frames_per_chunk, dtype=int)
                    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
                else:
                    frame_indices = [int((start_frame + end_frame)/2)]

                frames = vr.get_batch(frame_indices).asnumpy()
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2) # (16, C, H, W)
                chunk_inputs.append(frames_tensor)

            batch_input = torch.stack(chunk_inputs).to(device)

            with torch.no_grad():
                descriptions = captioner.describe(batch_input)
            
            total_dict[summary_key]['summary'] = descriptions

        except Exception as e:
            print(f"Error on {video_path}: {e}")
            total_dict[summary_key]['summary'] = ['NONE'] * num_chunks_per_video
        
        if (idx + 1) % save_interval == 0:
             with open(output_filename, "w") as json_file:
                json.dump(total_dict, json_file, indent=4)

    with open(output_filename, "w") as json_file:
        json.dump(total_dict, json_file, indent=4)
    
    print(f"GPU {args.gpu_id} Finished! Saved to {output_filename}")
