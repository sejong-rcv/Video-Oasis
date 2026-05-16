import sys
import os
import tqdm
import argparse
import json
import random
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModel, AutoConfig, AutoProcessor
from model import longclip

MODEL_NAMES = {
    "clip": "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-large-patch16-384",
    "siglip2": "google/siglip2-giant-opt-patch16-384",
    "eva": "BAAI/EVA-CLIP-8B",
    "longclip": "./checkpoints/LongCLIP-L/longclip-L.pt"
}

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    else:
        print("Not using DDP. Running on single GPU.")
        return 0, 0, 1

def get_sorted_video_list(anno, rank, model_id, output_folder):
    video_list = []
    unique_videos = set()
    
    for item in tqdm.tqdm(anno):
        if 'db' in item and 'video_path' in item:
            vidname = item['video_path'].split('/')[-1]
            # 이미 처리된 파일인지 체크
            save_path = os.path.join(output_folder, model_id, item['db'] + '**@@**' + vidname + '.pt')
            if not os.path.isfile(save_path):
                vid_key = item['db'] + '**@@**' + item['video_path']
                unique_videos.add(vid_key)
        
    unique_videos = list(unique_videos)
    unique_videos.sort()
    if rank == 0:
        print(f"Scanning file sizes for {len(unique_videos)} videos...")

    video_with_size = []
    iterator = tqdm.tqdm(unique_videos, disable=(rank != 0), desc="Checking File Sizes")
    
    for vid_key in iterator:
        vid_path = vid_key.split('**@@**')[1]
        try:
            size = os.stat(vid_path).st_size
            video_with_size.append((vid_key, size))
        except OSError:
            video_with_size.append((vid_key, -1))

    video_with_size.sort(key=lambda x: x[1], reverse=False)
    return [x[0] for x in video_with_size]

class VideoDataset(Dataset):
    def __init__(self, video_list, processor, is_longclip=False):
        self.video_list = video_list
        self.processor = processor
        self.is_longclip = is_longclip
        self.max_frames = 2048

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        vid_key = self.video_list[idx]
        db, vid_path = vid_key.split('**@@**')
        vid_name = vid_path.split('/')[-1]
        
        try:
            vr = VideoReader(vid_path, ctx=cpu(0), num_threads=1)
            fps = max(1, int(vr.get_avg_fps()))
            sample_idx = np.arange(0, len(vr), fps).astype(int)

            if len(sample_idx) > self.max_frames:
                sample_idx = np.linspace(0, len(vr) - 1, self.max_frames).astype(int)

            sample_idx = np.clip(sample_idx, 0, len(vr)-1)
            frames = vr.get_batch(sample_idx).asnumpy()  # (T, H, W, C)
            
            if self.is_longclip:
                pixel_values = torch.stack([self.processor(Image.fromarray(f)) for f in frames])
            else:
                inputs = self.processor(images=list(frames), return_tensors="pt", padding=True)
                pixel_values = inputs.pixel_values 

            return {"pixel_values": pixel_values, "db": db, "vid_name": vid_name, "success": True}
        except Exception as e:
            return {"db": db, "vid_name": vid_name, "error_msg": str(e), "success": False}

def collate_fn(batch):
    return batch[0]

def main():
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, default='/mnt/gtlim_data/users/gtlim/features/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_id', type=str, default='siglip', choices=MODEL_NAMES.keys())
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_folder, args.model_id), exist_ok=True)

    model_key = MODEL_NAMES[args.model_id]
    is_longclip = (args.model_id == 'longclip')

    # 모델 로드 분기
    if is_longclip:
        model, processor = longclip.load(model_key, device=device)
    elif args.model_id == 'eva':
        model = AutoModel.from_pretrained(model_key, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
        model = AutoModel.from_pretrained(model_key).to(device)
        processor = AutoProcessor.from_pretrained(model_key)
    
    model.eval()
    model = torch.compile(model)
    
    anno_path = "/mnt/users/gtlim/workspace/src/lmms_eval/vqa_total.json"
    anno = json.load(open(anno_path))
    
    sorted_video_list = get_sorted_video_list(anno, rank, args.model_id, args.output_folder)
    dataset = VideoDataset(sorted_video_list, processor, is_longclip=is_longclip)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)

    if rank == 0:
        print(f"Processing {args.model_id} features...")

    with torch.inference_mode():
        for batch_data in tqdm.tqdm(dataloader, disable=(rank != 0)):
            try:
                if not batch_data['success']:
                    if rank == 0: print(f"Error loading {batch_data['vid_name']}: {batch_data['error_msg']}")
                    continue

                db, vid_name = batch_data['db'], batch_data['vid_name']
                out_path = os.path.join(args.output_folder, args.model_id, f'{db}**@@**{vid_name}.pt')
                
                pixel_values = batch_data['pixel_values'].to(device)

                # Long-CLIP/EVA는 FP16을 주로 사용하므로 맞춰줌
                dtype = torch.float16 if (is_longclip or args.model_id == 'eva') else torch.bfloat16
                pixel_values = pixel_values.to(dtype)
                
                embeddings_list = []
                for i in range(0, pixel_values.shape[0], args.batch_size):
                    batch_frames = pixel_values[i : i + args.batch_size]
                    if  args.model_id == 'eva':
                        with torch.amp.autocast("cuda"):
                            feat = model.encode_image(batch_frames)
                    elif is_longclip:
                        feat = model.encode_image(batch_frames)
                    else:
                        feat = model.get_image_features(pixel_values=batch_frames)
                    embeddings_list.append(feat.cpu())

                if embeddings_list:
                    final_embedding = torch.cat(embeddings_list, dim=0)
                    torch.save(final_embedding, out_path)
            except Exception as e:
                print(e)
                continue
    if rank == 0: print("Processing Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    random.seed(42)
    main()