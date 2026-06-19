
import os
import json
import tqdm
import torch
import argparse

from transformers import AutoProcessor, AutoModel
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

parser = argparse.ArgumentParser(description="Eval Language Only")
parser.add_argument('--anno', type=str, default='/mnt/users/gtlim/workspace/src/lmms_eval/vqa_total.json')
parser.add_argument('--model_version', type=str, choices=['qwen25_vl','qwen3_vl','eagle25'])
args = parser.parse_args()

if args.model_version == 'qwen25_vl':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/mnt/gtlim_data/users/gtlim/models/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
    processor = AutoProcessor.from_pretrained("/mnt/gtlim_data/users/gtlim/models/Qwen2.5-VL-7B-Instruct")

if args.model_version == 'qwen3_vl':
    model = Qwen3VLForConditionalGeneration.from_pretrained("/mnt/gtlim_data/users/gtlim/models/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda",)
    processor = AutoProcessor.from_pretrained("/mnt/gtlim_data/users/gtlim/models/Qwen3-VL-8B-Instruct")

if args.model_version == 'eagle25':
    model = AutoModel.from_pretrained("/mnt/gtlim_data/users/gtlim/models/Eagle2.5-8B", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
    processor = AutoProcessor.from_pretrained("/mnt/gtlim_data/users/gtlim/models/Eagle2.5-8B", trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"

model.eval()


def get_answer(question, model_version):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    if model_version != 'eagle25':
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]


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


