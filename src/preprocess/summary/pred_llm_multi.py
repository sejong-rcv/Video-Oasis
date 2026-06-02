import os
import json
import tqdm
import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing as mp


def worker(gpu_id, sub_anno, summary_data, args):
    device = f"cuda:{gpu_id}"
    
    if args.model_version == 'llama':
        model = transformers.pipeline("text-generation", model='/mnt/gtlim_data/users/gtlim/models/Llama-3.1-8B-Instruct', model_kwargs={"torch_dtype": torch.bfloat16}, device=device)
    elif args.model_version == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Qwen3-8B')
        model = AutoModelForCausalLM.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Qwen3-8B', torch_dtype=torch.bfloat16).to(device)
        model.eval()
    elif args.model_version == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Mistral-7B-Instruct-v0.3')
        model = AutoModelForCausalLM.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Mistral-7B-Instruct-v0.3', torch_dtype=torch.bfloat16).to(device)
        model.eval()

    pbar = tqdm.tqdm(sub_anno, desc=f"GPU {gpu_id}")
    for ann in pbar:
        db = ann['db']
        qid = ann['qid']
        if '/' in qid:
            qid = qid.replace('/','_')
        save_path = os.path.join(f"./output_1frame/{args.model_version}", f"{db}**@@**{qid}.json")
        
        if os.path.isfile(save_path):
            continue

        try:
            option_prompt = "You are a helpful assistant. Select the best answer to the following multiple-choice question based on the question and options."
            question = ann['question']
            option = "\n".join(ann["options"])
            
            video_key = ann['video_path'].split('benchmark')[-1]
            video_summary_list = summary_data.get(video_key, {}).get('summary', [])
            narration = "".join([f"Scene Number #{i} : {video_summary_list[i]}\n\n" for i in range(len(video_summary_list))])

            cand = '(' + ", ".join([chr(i + 65) for i in range(len(ann['options'])-1)]) + f" or {chr(len(ann['options'])+64)})"
            full_prompt = f"{option_prompt}\n\nQuestion: {question}\n\nVideo Description:\n{narration}\nOptions:\n{option}\n\nRespond with only the letter {cand} of the correct option. Put your final answer in \\boxed{{}}."

            messages = [{"role": "user", "content": full_prompt}]
            
            if args.model_version == 'qwen':
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer([text], return_tensors="pt").to(device)
                generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                output_text = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            elif args.model_version == 'mistral':
                inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
                generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
                output_text = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            elif args.model_version == 'llama':
                res = model(messages, max_new_tokens=1024, do_sample=False)
                output_text = res[0]["generated_text"][-1]['content']

            ann['pred'] = output_text
            with open(save_path, "w") as f:
                json.dump(ann, f, indent=4)
        except Exception as e:
            print(f"Error at GPU {gpu_id}: {save_path} -> {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno', type=str, default='/mnt/users/gtlim/workspace/Video-Oasis/src/lmms_eval/video_total.json')
    parser.add_argument('--model_version', type=str, choices=['llama','qwen','mistral'])
    parser.add_argument('--num_gpus', type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(f"./output_1frame/{args.model_version}"):
        os.makedirs(f"./output_1frame/{args.model_version}", exist_ok=True)

    total_anno = json.load(open(args.anno))
    anno = []
    for ann in total_anno:
        db = ann['db']
        qid = ann['qid']
        if '/' in qid:
            qid = qid.replace('/','_')
        save_path = os.path.join(f"./output_1frame/{args.model_version}", f"{db}**@@**{qid}.json")
        if os.path.isfile(save_path)==False:
            anno.append(ann)

    summary_data = json.load(open('./total_summary_1frame.json'))

    num_gpus = args.num_gpus
    chunk_size = (len(anno) + num_gpus - 1) // num_gpus
    chunks = [anno[i:i + chunk_size] for i in range(0, len(anno), chunk_size)]

    mp.set_start_method('spawn', force=True)
    processes = []
    for i in range(num_gpus):
        if i < len(chunks):
            p = mp.Process(target=worker, args=(i, chunks[i], summary_data, args))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()