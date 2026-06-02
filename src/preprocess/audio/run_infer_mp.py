# for multiprocessing, 기존 run_inference 사실 지워도 될듯
# for Qwen2.5 VL, Qwen3 VL
import os
import json
import tqdm
import torch
import re
import argparse
import huggingface_hub
import torch.multiprocessing as mp
import math
from modules.data_loader import load_video_mme_data
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='video-mme',choices=os.listdir('/data3/jylee/workspace/Whisper/STT')) # dataset name
parser.add_argument('--model_version', type=str, default='qwen25_vl',choices=['qwen25_vl','qwen3_vl','eagle25'])    # model name 
parser.add_argument('--gpus', type=str, default='0,1', help='IDs of GPUs to use, comma separated (e.g. 0,1)')
args = parser.parse_args()

def format_transcript_with_timestamps(stt_data):
    if not stt_data or "timestamps" not in stt_data:
        return stt_data.get("transcript", "") if stt_data else "No transcript available."
    formatted_lines = []
    for chunk in stt_data["timestamps"]:
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        if start is None: start = 0.0
        if end is None: end = 0.0
        formatted_lines.append(f"[{start:.1f} - {end:.1f}] {text}")
    return "\n".join(formatted_lines)

def build_timestamp_prompt(stt_data, question, options):
    transcript_context = format_transcript_with_timestamps(stt_data)
    options_str = "\n".join(options)
    num_options = len(options)
    last_char = chr(ord('A') + num_options - 1) if num_options > 0 else "Z"
    valid_range = f"A to {last_char}"

    prompt = f"""You are an AI assistant tasked with answering questions based STRICTLY on the provided video transcript.

[INSTRUCTIONS]
1. Read the transcript with timestamps carefully.
2. Select the correct option ({valid_range}).
3. **CRITICAL**: You MUST cite the specific timestamp range (e.g., [12.5 - 15.0]) from the transcript that supports your answer.
4. **DO NOT GUESS**: If the provided transcript does not contain enough information to answer the question, acknowledge the uncertainty in the 'Reasoning' section, but still select the most plausible option based on context.
5. **OUTPUT FORMAT**: For the 'Answer:' field, write ONLY the option letter (e.g., 'Answer: A'). Do not add any other text or parentheses.
[Transcript]
{transcript_context}

[Question]
{question}

[Options]
{options_str}

[Format]
Answer: (Option Letter)
Evidence: (Quote the text and timestamp. If none, write "None")
Reasoning: (Explain why you chose this option. If you are guessing, explicitly state "Information missing, guessing based on context.")"""
    return prompt.strip()

def parse_prediction(pred_text):
    if not pred_text: return "Unknown", "", ""
    match_option = re.search(r"Answer:\s*\(?([A-D])\)?", pred_text, re.IGNORECASE)
    pred_option = match_option.group(1).upper() if match_option else "Unknown"

    evidence, reasoning = "", ""
    if "Evidence:" in pred_text:
        parts = pred_text.split("Evidence:", 1)[1]
        if "Reasoning:" in parts:
            evidence = parts.split("Reasoning:", 1)[0].strip()
            reasoning = parts.split("Reasoning:", 1)[1].strip()
        else:
            evidence = parts.strip()
    
    return pred_option, evidence, reasoning

# --- [Inference Function] (모델 인스턴스를 인자로 받도록 수정) ---

def get_answer_open(model, processor, question, model_version, device):
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text], images=None, videos=None, padding=True, return_tensors="pt"
    ).to(device)

    # Truncation (VRAM 폭발 방지)
    # MAX_LEN = 16000
    # if inputs.input_ids.shape[1] > MAX_LEN:
    #     inputs.input_ids = inputs.input_ids[:, :MAX_LEN]
    #     inputs.attention_mask = inputs.attention_mask[:, :MAX_LEN]

    gen_kwargs = {"max_new_tokens": 1024, "do_sample": False, "temperature": 0.0}
    
    # 모델별 분기
    if model_version != 'eagle25':
        generated_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    else:
        generated_ids = model.generate(**inputs, **gen_kwargs)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return output_text[0]

# --- [Worker Process] ---

def gpu_worker(rank, gpu_ids, args, data_chunk):
    """
    각 GPU 프로세스에서 실행될 함수
    """
    gpu_id = gpu_ids[rank]
    device = f"cuda:{gpu_id}"
    print(f"🚀 [Worker {rank}] Start Inference on GPU {gpu_id} | Data Size: {len(data_chunk)}")
    
    # 1. 모델 로드 (프로세스별로 개별 로드)
    # BASE_MODEL_PATH 경로 수정 필요시 수정
    
    model = None
    processor = None

    try:
        if args.model_version == 'qwen25_vl':
            model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_path)
            
        elif args.model_version == 'eagle25':
            # Eagle 2.5 로컬 경로 또는 ID
            model_path = "nvidia/Eagle2.5-8B" 
            # 로컬 경로 예시: model_path = "/data3/sjpark/workspace/LVU_release/models/Eagle2.5-8B"
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2", 
            ).to(device)
            
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.padding_side = "left"

        elif args.model_version == 'qwen3_vl':
            model_path = "Qwen/Qwen3-VL-8B-Instruct"
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            
        model.eval()
        
    except Exception as e:
        print(f"❌ [Worker {rank}] Model Load Error: {e}")
        return

    # 2. 추론 루프
    results = []
    STT_PATH = os.path.join('/data3/jylee/workspace/Whisper/STT_clean',f"{args.dataset}") # change

    for item in tqdm.tqdm(data_chunk, desc=f"Worker {rank}", position=rank):
        # video id 파싱 
        video_id = item['video']
        if args.dataset == 'vcr-bench':
            video_id = video_id.split('/')[-1]
        if args.dataset == 'longvideobench' and '@' in video_id :
            video_id = video_id.split('-')[-1]
        stt_file = os.path.join(STT_PATH, f"{video_id}.mp3.json")

        stt_data = None
        if os.path.exists(stt_file):
            try:
                with open(stt_file, 'r', encoding='utf-8') as f:
                    stt_data = json.load(f)
            except: pass
        else : 
            continue
        
        try:
            prompt = build_timestamp_prompt(stt_data, item['question'], item['options'])
            pred = get_answer_open(model, processor, prompt, args.model_version, device)
            pred_option, evidence, reasoning = parse_prediction(pred)
            results.append({
                "qid": item['qid'],
                "video": item['video'].replace('.mp3', ''),
                "db": item.get('db', 'Video-MME'),
                "question": item['question'],
                "options": item['options'],
                "ground_truth": item['answer'],
                "predicted_option": pred_option,
                "is_correct": (item['answer'] == pred_option),
                "Evidence": evidence,
                "Reasoning": reasoning,
            })
        except Exception as e:
            print(f"⚠️ Error on {item['qid']}: {e}")
        
    # 3. 부분 결과 저장
    save_dir = f"./output_prior/{args.model_version}" # change
    os.makedirs(save_dir, exist_ok=True)
    part_file = os.path.join(save_dir, f"part_{rank}.json")
    
    with open(part_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ [Worker {rank}] Finished! Saved to {part_file}")

# --- [Main Entry] ---

if __name__ == '__main__':
    huggingface_hub.login(token='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') # 토큰 확인
    mp.set_start_method('spawn', force=True) # CUDA 멀티프로세싱 필수

    # 1. GPU 설정
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"🔥 Using {num_gpus} GPUs: {gpu_ids}")

    # 2. 데이터 로드 및 분할
    JSON_PATH = '/data3/jylee/workspace/Whisper/vqa_total.json'
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # vqa_total의 db 이름 매칭
    db_list = ['LongVideoBench', 'MLVU_Test', 'MMR-V','MVBench','RTV-Bench','TVBench','VCR-Bench','Video-Holmes','Video-MME']
    db_list.sort()
    bench_dict = {k:v for v,k in zip(db_list,sorted(os.listdir('/data3/jylee/workspace/Whisper/STT'))) }
    target_data = [
        item for item in all_data 
        if  item.get('db') == bench_dict[args.dataset]
    ]
    
    if len(target_data) == 0:
        print(f"⚠️ Warning: '{args.dataset}'에 해당하는 데이터를 찾지 못했습니다. 종료합니다.")
        exit()
    
    print(f"📂 Total Data: {len(target_data)}")
    
    # 데이터 쪼개기 (Chunking)
    chunk_size = math.ceil(len(target_data) / num_gpus)
    data_chunks = [target_data[i:i + chunk_size] for i in range(0, len(target_data), chunk_size)]

    # 디버깅용 코드
    # print("🐞 Debugging Mode: Running Single Process...") 
    # debug_chunk = data_chunks[0][:10] 
    
    # # 프로세스 생성(mp.Process) 대신 함수 직접 호출
    # # rank=0, gpu_ids=[0번GPU], args=args, data=debug_chunk
    # gpu_worker(0, [gpu_ids[0]], args, debug_chunk)
    
    # print("✅ Debugging Finished")
    # exit() # 여기서 끝냄

    # 3. 멀티프로세스 실행
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(rank, gpu_ids, args, data_chunks[rank]))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    # 4. 결과 병합 (Merge)
    print("🔄 Merging results...")
    final_results = []
    save_dir = f"./output_prior/{args.model_version}" # change
    
    for rank in range(num_gpus):
        part_file = os.path.join(save_dir, f"part_{rank}.json")
        if os.path.exists(part_file):
            with open(part_file, 'r', encoding='utf-8') as f:
                final_results.extend(json.load(f))
            os.remove(part_file) # 병합 후 부분 파일 삭제 (선택사항)

    final_output = os.path.join(save_dir, f"{args.dataset}_result.json")
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"🏆 All Done! Final results saved to: {final_output}")
    # Execute : OMP_NUM_THREADS=1 python run_infer_mp.py --gpus "0,1" --dataset "rtv-bench" --model_version "qwen25_vl"