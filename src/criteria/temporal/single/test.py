import argparse
import json
import os

import torch
import tqdm
from decord import VideoReader, cpu
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


MODEL_CHOICES = [
    "qwen25_vl",
    "qwen3_vl",
    "eagle25",
]


def load_model(model_version):
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


def resolve_video_path(video_path):
    return video_path.replace(
        "../data/benchmarks",
        "/data3/gtlim/workspace/src/Video-Oasis/data/benchmarks",
    )


def load_center_frame(ann):
    video_path = resolve_video_path(ann["video_path"])
    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(video_reader)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    if ann["db"] == "RTV-Bench":
        fps = float(video_reader.get_avg_fps())
        start_time = float(ann["start_time"])
        end_time = float(ann["end_time"])
        center_time = (start_time + end_time) / 2
        frame_index = round(center_time * fps)
    else:
        frame_index = total_frames // 2

    frame_index = max(0, min(frame_index, total_frames - 1))
    frame = video_reader[frame_index].asnumpy()
    return Image.fromarray(frame).convert("RGB")


def get_answer(question, frame, model_version, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[frame],
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


def build_prompt(ann):
    options = ann["options"]
    if not options:
        raise ValueError("The sample has no answer options.")

    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    if len(option_letters) == 1:
        candidates = f"({option_letters[0]})"
    else:
        candidates = f"({', '.join(option_letters[:-1])}, or {option_letters[-1]})"

    option_text = "\n".join(options)
    return (
        "You are a helpful assistant. Select the best answer to the following "
        "multiple-choice question based on the image, question, and options.\n\n"
        f"Question:\n{ann['question']}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Respond with only the letter {candidates} of the correct option. "
        "Put your final answer in \\boxed{}."
    )


def main():
    parser = argparse.ArgumentParser(description="Run single-frame inference")
    parser.add_argument(
        "--anno",
        type=str,
        default="/data3/gtlim/workspace/src/Video-Oasis/src/lmms_eval/video_total.json",
    )
    parser.add_argument("--model_version", required=True, choices=MODEL_CHOICES)
    args = parser.parse_args()

    model, processor = load_model(args.model_version)
    output_dir = os.path.join("output", args.model_version)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.anno, "r") as f:
        annotations = json.load(f)

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
            frame = load_center_frame(ann)
            ann["pred"] = get_answer(
                build_prompt(ann),
                frame,
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
