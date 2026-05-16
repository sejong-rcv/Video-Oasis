import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

TASK_CATEGORIES = ['egoschema', 'ImplicitQA','LongVideoBench','LVBench','MINERVA','MLVU_Test','MMR-V','MVBench','RTV-Bench','TVBench','VCR-Bench','Video-Holmes','Video-MME','VSI-Bench']

replace_prompt = " Please answer yes or no."

with open(Path(__file__).parent / "vqa_total.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

def vqa_total_doc_to_visual(doc):
    video_path = doc["video_path"]
    return [video_path]

def vqa_total_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # 여기서 option prompt 질문마다 다르게 주자
    cand = '('
    for i in range(len(doc['options'])):
        if i == len(doc['options'])-1:
            cand += 'or ' + chr(i + 65) + ')'
        else:
            cand += chr(i + 65) + ', '

    option_prompt = f"Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter {cand} of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFGHIJKLNM]", s):
        return ""

    matches = re.search(r"[ABCDEFGHIJKLNM]", s)
    if matches is None:
        return ""
    return matches[0]

def vqa_total_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case vqa_total score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    data_dict = {"question_id": doc["qid"], "task_type": doc["db"], "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"vqa_total_perception_score": data_dict}


def vqa_total_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for task_category in TASK_CATEGORIES:
        category2score[task_category] = {"correct": 0, "answered": 0}

    for result in results:
        task_category = result["task_type"]
        category2score[task_category]["answered"] += 1
        category2score[task_category]["correct"] += result["pred_answer"] == result["answer"]

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
