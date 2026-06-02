import os
import json
import tqdm
import torch
import argparse
import multiprocessing as mp
import random

from transformers import AutoProcessor, AutoModel
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

# ==========================================
# 1. 핵심 추론 함수 (각 GPU 프로세스가 실행할 함수)
# ==========================================
def worker(gpu_id, total_gpus, args, my_anno, summary_data):
    # 각 프로세스별 GPU 설정
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id) # 현재 프로세스의 기본 GPU 설정

    print(f"📡 [GPU {gpu_id}] Loading model... (Task: {len(my_anno)} items)")

    # 모델 로드
    if args.model_version == 'qwen25_vl':
        model_path = "/mnt/gtlim_data/users/gtlim/models/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif args.model_version == 'qwen3_vl':
        model_path = "/mnt/gtlim_data/users/gtlim/models/Qwen3-VL-8B-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif args.model_version == 'eagle25':
        model_path = "/mnt/gtlim_data/users/gtlim/models/Eagle2.5-8B"
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        processor.tokenizer.padding_side = "left"

    model.eval()

    # 결과 폴더
    output_base_dir = f"./{args.output_dir}/{args.model_version}"

    # 해당 GPU 전용 tqdm (position을 gpu_id로 주면 바(bar)가 겹치지 않고 정렬됨)
    for ann in tqdm.tqdm(my_anno, desc=f"GPU {gpu_id}", position=gpu_id, leave=True):
        db = ann['db']
        qid = ann['qid']
        save_path = os.path.join(output_base_dir, f"{db}**@@**{qid}.json")
        
        if not os.path.isfile(save_path):
            try:
                option_prompt = "You are a helpful assistant. Select the best answer to the following multiple-choice question..."
                question_text = ann['question']
                options = "\n".join(ann["options"])
                
                # Summary 결합
                narration = ""
                video_summary_list = summary_data[ann['video_path']]['summary']

                if args.output_dir.split('_')[-1] == "shuffled":
                    random.shuffle(video_summary_list)

                for i in range(len(video_summary_list)):
                    narration += f"Scene Number #{i} : {video_summary_list[i]}\n\n"
                
                full_prompt = f"{option_prompt}\n\nQuestion: {question_text}\n\nVideo Description:\n{narration}\nOptions:\n{options}\n\nRespond with only the letter..."
                
                # Inference
                messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=None, videos=None, padding=True, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                    if args.model_version != 'eagle25':
                        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                ann['pred'] = pred
                with open(save_path, "w") as json_file:
                    json.dump(ann, json_file, indent=4)
                    
            except Exception as e:
                pass # 에러 처리는 기존 방식 유지

# ==========================================
# 2. 메인 실행부 (데이터 분할 및 프로세스 생성)
# ==========================================
if __name__ == '__main__':
    # [중요] CUDA 사용 시 자식 프로세스를 생성하는 가장 안전한 방법
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 이미 설정된 경우 에러 방지
        pass    

    parser = argparse.ArgumentParser(description="Multi-GPU Automated Parallel Eval")
    parser.add_argument('--anno', type=str, default='/mnt/users/gtlim/workspace/Video-Oasis/src/lmms_eval/video_total.json')
    parser.add_argument('--output_dir', type=str, choices=['output_1frame_ordered','output_1frame_shuffled','output_16frame_ordered','output_16frame_shuffled'])
    parser.add_argument('--summary_file', type=str, choices=['total_summary_1frame.json','total_summary_16frame.json'])
    parser.add_argument('--model_version', type=str, choices=['qwen25_vl','qwen3_vl','eagle25'])
    parser.add_argument('--total_gpus', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(f"./{args.output_dir}/{args.model_version}", exist_ok=True)
    random.seed(42)

    # 전체 데이터 로드
    anno = json.load(open(args.anno))
    summary_data = json.load(open(args.summary_file))

    # 데이터 분할 로직
    total_items = len(anno)
    chunk_size = total_items // args.total_gpus
    remainder = total_items % args.total_gpus

    processes = []
    
    for i in range(args.total_gpus):
        start_idx = i * chunk_size + min(i, remainder)
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        my_anno_chunk = anno[start_idx:end_idx]

        # 각 GPU에 맞는 프로세스 생성
        p = mp.Process(target=worker, args=(i, args.total_gpus, args, my_anno_chunk, summary_data))
        p.start()
        processes.append(p)

    # 모든 프로세스가 끝날 때까지 대기
    for p in processes:
        p.join()

    print("🏁 All GPUs have finished their tasks!")