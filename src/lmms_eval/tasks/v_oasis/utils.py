import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.eval_utils import extract_answer_letter
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

TASK_CATEGORIES = ['egoschema', 'ImplicitQA','LongVideoBench','LVBench','MINERVA','MLVU_Test','MMR-V','MVBench','RTV-Bench','TVBench','VCR-Bench','Video-Holmes','Video-MME','VSI-Bench']
OASIS_CATEGORIES = [
    "A. Temporal Dynamics & Tracking",
    "B. Spatial World Understanding",
    "C. Causality & Logical Reasoning",
    "D. Global Narrative & Long-Term Context",
    "E. Fine-Grained Perception",
]

replace_prompt = " Please answer yes or no."

with open(Path(__file__).parent / "v_oasis.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

def v_oasis_doc_to_visual(doc):
    if doc['db']=='RTV-Bench':
        video_path = doc["video_path"]
        return [(video_path, doc["start_time"], doc["end_time"])]
    else:
        video_path = doc["video_path"]
        return [video_path]

def v_oasis_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    cand = '('
    for i in range(len(doc['options'])):
        if i == len(doc['options'])-1:
            cand += 'or ' + chr(i + 65) + ')'
        else:
            cand += chr(i + 65) + ', '

    option_prompt = f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter {cand} of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt

    return full_prompt

def extract_characters_regex(s, num_options=None):
    return extract_answer_letter(s, num_options)

def v_oasis_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case v_oasis score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred, len(doc.get("options") or []))
    data_dict = {
        "question_id": doc["qid"],
        "task_type": doc["db"],
        "oasis_category": doc["oasis_category"],
        "pred_answer": pred_ans,
        "answer": doc["answer"],
    }

    return {f"v_oasis_perception_score": data_dict}


def v_oasis_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    task_scores = {category: {"correct": 0, "answered": 0} for category in TASK_CATEGORIES}
    oasis_scores = {category: {"correct": 0, "answered": 0} for category in OASIS_CATEGORIES}

    for result in results:
        task_category = result["task_type"]
        oasis_category = result["oasis_category"]
        is_correct = result["pred_answer"] == result["answer"]

        if task_category not in task_scores:
            task_scores[task_category] = {"correct": 0, "answered": 0}
        if oasis_category not in oasis_scores:
            oasis_scores[oasis_category] = {"correct": 0, "answered": 0}

        task_scores[task_category]["answered"] += 1
        task_scores[task_category]["correct"] += is_correct
        oasis_scores[oasis_category]["answered"] += 1
        oasis_scores[oasis_category]["correct"] += is_correct

    for category, score in oasis_scores.items():
        accuracy = 100 * score["correct"] / score["answered"] if score["answered"] else 0
        eval_logger.info(f"Evaluation on OASIS Categories: {category}: {accuracy:.1f}%")

    for category, score in task_scores.items():
        accuracy = 100 * score["correct"] / score["answered"] if score["answered"] else 0
        eval_logger.info(f"Evaluation on Source Benchmarks: {category}: {accuracy:.1f}%")

    total_correct = sum(score["correct"] for score in oasis_scores.values())
    total_answered = sum(score["answered"] for score in oasis_scores.values())
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
