import os
import json
import tqdm
import re

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

def get_db_wise_dict(preds):
    output = dict()
    for pred in preds:
        if pred['db'] not in output.keys():
            output[pred['db']] = dict()
            output[pred['db']]['correct_cnt'] = 0
            output[pred['db']]['valid_cnt'] = 0
    output['Total'] = dict()
    output['Total']['correct_cnt'] = 0
    output['Total']['valid_cnt'] = 0

    return output

def get_criteria_wise_dict():
    output = dict()
    for group in ['spatial','temporal','causal','global','static','overall']:
        output[group] = dict()
        output[group]['correct_cnt'] = 0
        output[group]['valid_cnt'] = 0
    return output

if __name__ == '__main__':

    missing_cnt = 0 

    input_file = '/mnt/users/gtlim/workspace/Video-Oasis/src/experiments/qwen_benchmark/Qwen3.5-27B/min16_max256_total128000_maxf128_sampling/models__Qwen3.5-27B/20260428_141534_samples_v_oasis.jsonl'

    total_anno = json.load(open("/mnt/users/gtlim/workspace/Video-Oasis/src/lmms_eval/video_total.json"))
    oasis_anno = json.load(open("/mnt/users/gtlim/workspace/Video-Oasis/src/lmms_eval/video_oasis.json"))

    if input_file.endswith('.jsonl'):
        preds = []
        mode = 'jsonl'
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                preds.append(json.loads(line))
    else:
        mode = 'json'
        preds = json.load(open(input_file))
        for pred in preds:
            if 'processing_time' in pred.keys(): # videotree
                try:
                    pred['pred'] = chr(pred['pred'] + 65)
                except:
                    pred['pred'] = ''
                pred['answer'] = chr(pred['truth'] + 65)

            if 'quid' in pred.keys(): # videotool
                try:
                    pred['pred'] = chr(pred['final_predicted_option'] + 65)
                except:
                    pred['pred'] = ''
                pred['answer'] = chr(pred['truth'] + 65)
            elif '<answer>' in pred['pred']:
                pred['pred'] = pred['pred'].split('<answer>')[-1][0]

    db_dict = get_db_wise_dict(total_anno)
    criteria_dict = get_criteria_wise_dict()

    preds_key = []
    for pred in preds:
        if mode == 'jsonl':
            key = pred['v_oasis_perception_score']['task_type'] + '**@@**' + pred['v_oasis_perception_score']['question_id']
        else:
            try:
                key = pred['db'] + '**@@**' + pred['qid']
            except:
                key = pred['vid_key']
        preds_key.append(key)

    oasis_key = []
    spatial_set = []
    temporal_set = []
    causal_set = []
    global_set = []
    static_set = []

    for ann in oasis_anno:
        key = ann['db'] + '**@@**' + ann['qid']
        if ann['oasis_category'] == 'A. Temporal Dynamics & Tracking':
            temporal_set.append(key)
        if ann['oasis_category'] == 'B. Spatial World Understanding':
            spatial_set.append(key)
        if ann['oasis_category'] == 'C. Causality & Logical Reasoning':
            causal_set.append(key)
        if ann['oasis_category'] == 'D. Global Narrative & Long-Term Context':
            global_set.append(key)
        if ann['oasis_category'] == 'E. Static Perception & Retrieval':
            static_set.append(key)
        oasis_key.append(key)

    for key in oasis_key:
        if key not in preds_key:
            criteria_dict['overall']['valid_cnt'] += 1
            if oasis_key in spatial_set:
                criteria_dict['spatial']['valid_cnt'] += 1
            if oasis_key in temporal_set:
                criteria_dict['temporal']['valid_cnt'] += 1
            if oasis_key in causal_set:
                criteria_dict['causal']['valid_cnt'] += 1
            if oasis_key in global_set:
                criteria_dict['global']['valid_cnt'] += 1
            if oasis_key in static_set:
                criteria_dict['static']['valid_cnt'] += 1

    candidates = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    for anno in tqdm.tqdm(preds):
        if 'vid_key' in anno.keys():
            key = anno['vid_key']
            anno['answer'] = chr(65 + anno['truth'])
        elif mode == 'json':
            key = anno['db'] + '**@@**' + anno['qid']
        elif mode == 'jsonl':
            key = anno['v_oasis_perception_score']['task_type'] + '**@@**' + anno['v_oasis_perception_score']['question_id']

        if key not in oasis_key:
            continue

        if mode == 'jsonl':
            pred = anno['filtered_resps']
            answer = anno['v_oasis_perception_score']['answer']
            db = anno['v_oasis_perception_score']['task_type']

        if db =='videomme':
            db = 'Video-MME'

        db_dict['Total']['valid_cnt'] += 1
        db_dict[db]['valid_cnt'] += 1
        if key in oasis_key:
            criteria_dict['overall']['valid_cnt'] += 1
            if key in spatial_set:
                criteria_dict['spatial']['valid_cnt'] += 1
            if key in temporal_set:
                criteria_dict['temporal']['valid_cnt'] += 1
            if key in causal_set:
                criteria_dict['causal']['valid_cnt'] += 1
            if key in global_set:
                criteria_dict['global']['valid_cnt'] += 1
            if key in static_set:
                criteria_dict['static']['valid_cnt'] += 1

        if len(pred)==0:
            continue

        if type(pred) == int:
            pred = chr(65 + pred)
        else:
            if pred[0] in candidates:
                pred = pred[0]
            elif "boxed" in pred[0]:
                pred = pred[0].split("boxed{")[-1][0]
            else:
                pred = extract_characters_regex(pred[0])

        if pred == answer:
            db_dict['Total']['correct_cnt'] += 1
            db_dict[db]['correct_cnt'] += 1
            if key in oasis_key:
                criteria_dict['overall']['correct_cnt'] += 1
                if key in spatial_set:
                    criteria_dict['spatial']['correct_cnt'] += 1
                if key in temporal_set:
                    criteria_dict['temporal']['correct_cnt'] += 1
                if key in causal_set:
                    criteria_dict['causal']['correct_cnt'] += 1
                if key in global_set:
                    criteria_dict['global']['correct_cnt'] += 1
                if key in static_set:
                    criteria_dict['static']['correct_cnt'] += 1

    for db in ['egoschema', 'ImplicitQA', 'MINERVA', 'RTV-Bench', 'VSI-Bench', 'MVBench', 'LongVideoBench', 'LVBench', 'MLVU_Test', 'MMR-V', 'TVBench','VCR-Bench', 'Video-Holmes', 'Video-MME', 'Total']:
        print("DB : {:15s} || ACC : {:.2f}".format(db, 100*db_dict[db]['correct_cnt'] / db_dict[db]['valid_cnt']))

    for criteria in criteria_dict.keys():
        print("DB : {:15s} || ACC : {:.2f}".format(criteria, 100*criteria_dict[criteria]['correct_cnt'] / criteria_dict[criteria]['valid_cnt']))

    import pdb;pdb.set_trace()