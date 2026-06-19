
import os
import json
import tqdm
import torch
import argparse
import transformers
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

parser = argparse.ArgumentParser(description="Eval Language Only")
parser.add_argument('--anno', type=str, default='/mnt/users/gtlim/workspace/src/lmms_eval/vqa_total.json')
parser.add_argument('--model_version', type=str, choices=['llama','qwen','mistral'])
args = parser.parse_args()

if args.model_version == 'llama':
    model = transformers.pipeline("text-generation", model='/mnt/gtlim_data/users/gtlim/models/Llama-3.1-8B-Instruct', model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda")

if args.model_version == 'qwen':
    tokenizer = AutoTokenizer.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Qwen3-8B')
    model = AutoModelForCausalLM.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Qwen3-8B', torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

if args.model_version == 'mistral':
    tokenizer = AutoTokenizer.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Mistral-7B-Instruct-v0.3')
    model = AutoModelForCausalLM.from_pretrained('/mnt/gtlim_data/users/gtlim/models/Mistral-7B-Instruct-v0.3', torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

def get_answer(question, model_version):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    if model_version=='qwen':
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text=[text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    elif model_version=='mistral':
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    elif model_version=='llama':
        output_text = model(messages, max_new_tokens=1024, do_sample=False)
        output_text = output_text[0]["generated_text"][-1]

    return output_text


if __name__ == '__main__':
        
    if os.path.isdir(f"./output/{args.model_version}")==False:
        os.makedirs(f"./output/{args.model_version}", exist_ok=True)

    anno = json.load(open(args.anno))

    for ann in tqdm.tqdm(anno):
        db = ann['db']
        qid = ann['qid']
        if os.path.isfile(os.path.join(f"./output/{args.model_version}", f"{db}**@@**{qid}.json")) == False:
            try:
                option_prompt = "You are a helpful assistant. Select the best answer to the following multiple-choice question based on the question and options."
                question = ann['question']
                option = "\n".join(ann["options"])
                question = question + "\n" + option
                full_prompt = option_prompt + "\n"  + question + "\n" + "Respond with only the letter (A, B, C, D, E, F, G, H, I, J, K, L) of the correct option. Put your final answer in \\boxed{}."
                pred = get_answer(full_prompt, args.model_version)
                ann['pred'] = pred
                with open(os.path.join(f"./output/{args.model_version}", f"{db}**@@**{qid}.json"), "w") as json_file:
                    json.dump(ann, json_file, indent=4)
            except Exception as e:
                print(f"./output/{args.model_version}", f"{db}**@@**{qid}.json", e)


