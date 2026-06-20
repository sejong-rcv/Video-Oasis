import argparse
import json
import os

import torch
import tqdm
import transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


MODEL_CHOICES = [
    "llama",
    "qwen",
    "mistral",
    "qwen25_vl",
    "qwen3_vl",
    "eagle25",
]


def load_model(model_version):
    if model_version == "llama":
        model = transformers.pipeline(
            "text-generation",
            model="/data3/gtlim/workspace/src/Video-Oasis/data/models/Llama-3.1-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
        )
        return model, None

    if model_version == "qwen":
        model_path = "/data3/gtlim/workspace/src/Video-Oasis/data/models/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
        return model, tokenizer

    if model_version == "mistral":
        model_path = "/data3/gtlim/workspace/src/Video-Oasis/data/models/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
        return model, tokenizer

    if model_version == "qwen25_vl":
        model_path = "/data3/gtlim/workspace/src/Video-Oasis/data/models/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif model_version == "qwen3_vl":
        model_path = "/data3/gtlim/workspace/src/Video-Oasis/data/models/Qwen3-VL-8B-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_path)

    else:
        model_path = "/data3/gtlim/workspace/src/Video-Oasis/data/models/Eagle2.5-8B"
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        processor.tokenizer.padding_side = "left"

    model.eval()
    return model, processor


def get_answer(question, model_version, model, processor):
    if model_version == "llama":
        messages = [{"role": "user", "content": question}]
        output = model(
            messages,
            max_new_tokens=1024,
            do_sample=False,
        )
        return output[0]["generated_text"][-1]["content"].strip()

    if model_version in {"qwen", "mistral"}:
        messages = [{"role": "user", "content": question}]

        if model_version == "qwen":
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(text=[text], return_tensors="pt").to(model.device)
        else:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )
        output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
        return processor.decode(output_ids, skip_special_tokens=True).strip("\n")

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )
    if model_version != "eagle25":
        generated_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def get_video_summary(ann, summary_data):
    video_path = ann["video_path"]
    summary_key = video_path.replace("../data/benchmarks", "")
    summary_item = summary_data.get(summary_key)

    if summary_item is None:
        raise KeyError(f"Summary not found for video: {summary_key}")

    summaries = summary_item.get("summary", [])
    if not summaries:
        raise ValueError(f"Summary is empty for video: {video_path}")

    return summaries


def build_prompt(ann, summaries):
    options = ann["options"]
    if not options:
        raise ValueError("The sample has no answer options.")

    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    if len(option_letters) == 1:
        candidates = f"({option_letters[0]})"
    else:
        candidates = f"({', '.join(option_letters[:-1])}, or {option_letters[-1]})"

    narration = "".join(
        f"Scene Number #{i}: {summary}\n\n"
        for i, summary in enumerate(summaries)
    )
    option_text = "\n".join(options)

    return (
        "You are a helpful assistant. Select the best answer to the following "
        "multiple-choice question based on the video description, question, "
        "and options.\n\n"
        f"Question:\n{ann['question']}\n\n"
        f"Video Description:\n{narration}"
        f"Options:\n{option_text}\n\n"
        f"Respond with only the letter {candidates} of the correct option. "
        "Put your final answer in \\boxed{}."
    )


def main():
    parser = argparse.ArgumentParser(description="Run video-summary inference")
    parser.add_argument(
        "--anno",
        type=str,
        default="/data3/gtlim/workspace/src/Video-Oasis/src/lmms_eval/video_total.json",
    )
    parser.add_argument("--summary_file", type=str, required=True)
    parser.add_argument("--model_version", required=True, choices=MODEL_CHOICES)
    args = parser.parse_args()

    model, processor = load_model(args.model_version)
    output_dir = os.path.join("output", args.model_version)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.anno, "r") as f:
        annotations = json.load(f)
    with open(args.summary_file, "r") as f:
        summary_data = json.load(f)

    for ann in tqdm.tqdm(annotations):
        safe_qid = str(ann["qid"]).replace("/", "_")
        filename = f"{ann['db']}**@@**{safe_qid}.json"
        output_path = os.path.join(output_dir, filename)
        if os.path.isfile(output_path):
            with open(output_path, "r") as f:
                saved_result = json.load(f)
            if saved_result.get("pred"):
                continue

        try:
            summaries = get_video_summary(ann, summary_data)
            ann["pred"] = get_answer(
                build_prompt(ann, summaries),
                args.model_version,
                model,
                processor,
            )
            with open(output_path, "w") as f:
                json.dump(ann, f, indent=4)
        except Exception as e:
            print(output_path, e)


if __name__ == "__main__":
    main()
