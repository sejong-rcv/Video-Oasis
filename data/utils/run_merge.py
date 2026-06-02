import os
import json
import pandas as pd
import tqdm
import shutil
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NpEncoder, self).default(obj)

def get_files_by_extension(root_path):
    target_extensions = ('.json', '.jsonl', '.parquet', '.tsv')
    file_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(target_extensions):
                full_path = os.path.join(root, file)
                absolute_path = os.path.abspath(full_path)
                file_list.append(absolute_path)
    return file_list

if __name__ == '__main__':
    buffer = []
    root_dir = "/mnt/users/gtlim/workspace/src/benchmark/annos"
    result_files = get_files_by_extension(root_dir)
    Total_QA = dict()
    cnt = 0
    for file_path in tqdm.tqdm(result_files):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        elif ext == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))

        elif ext == '.parquet':
            data = pd.read_parquet(file_path, engine='pyarrow')

        elif ext == '.tsv':
            data = pd.read_csv(file_path, sep='\t')

        db = file_path.split('/')[8]
        if db not in Total_QA.keys():
            Total_QA[db] = dict()

        if db == 'ImplicitQA':
            for sample in data:
                option_txt = ''
                options = []
                for idx in sample['options'].keys():
                    option_txt += idx + '. ' + sample['options'][idx] + '. '
                    options.append(idx + '. ' + sample['options'][idx])
                    if idx == sample['answer_choice']:
                        answer_txt = sample['options'][idx]
                meta_info = f'Options of QA : {option_txt}\nCategory : {sample['category']}'
                Total_QA[db][sample['question_id']] = {
                    "question" : sample['question_text'],
                    "video" : sample['video_url'].split('v=')[-1],
                    "options" : options,
                    "answer" : sample['answer_choice'],
                    "answer_text" : answer_txt,
                    "meta" : meta_info,
                }

        elif db == 'LVBench':
            for sample in data:
                for item in sample['qa']:
                    option_txt = ''
                    options = []
                    for idx, opt in enumerate(item['question'].split('\n')[1:]):
                        option_txt += chr(idx + 65) + '. ' + opt.split(') ')[-1] + '. '
                        options.append(chr(idx + 65) + '. ' + opt.split(') ')[-1])
                        if chr(idx + 65) == item['answer']:
                            answer_txt = opt.split(') ')[-1]

                    meta_info = f'Options of QA : {option_txt}\nType : {". ".join(item['question_type'])} \nGenre : {sample['type']}\nDuration (min) : {sample['video_info']['duration_minutes']}'
                    Total_QA[db][item['uid']] = {
                        "question" : item['question'].split('\n')[0],
                        "video" : sample['key'],
                        "options" : options,
                        "answer" : item['answer'],
                        "answer_text" : answer_txt,
                        "meta" : meta_info,
                    }

        elif db == 'LongVideoBench':
            for sample in data:
                option_txt = ''
                for idx, opt in enumerate(sample['candidates']):
                    option_txt += chr(idx + 65) + '. ' + opt + '. '
                    sample['candidates'][idx] = chr(idx + 65) + '. ' + opt
                    if idx == sample['correct_choice']:
                        answer_txt = opt
                meta_info = f'Options of QA : {option_txt}\nTopic : {sample['topic_category']}\nDuration (sec) : {sample['duration']}'
                Total_QA[db][sample['id']] = {
                    "question" : sample['question'],
                    "video" : sample['video_id'],
                    "options" : sample['candidates'],
                    "answer" : chr(sample['correct_choice'] + 65),
                    "answer_text" : answer_txt,
                    "meta" : meta_info,
                }

        elif db == 'MLVU_Test':
            for sample in data:
                option_txt = ''
                for idx, opt in enumerate(sample['candidates']):
                    option_txt += chr(idx + 65) + '. ' + str(opt) + '. '
                    sample['candidates'][idx] = chr(idx + 65) + '. ' + str(opt)
                    if opt == sample['answer']:
                        answer = chr(idx + 65)
                meta_info = f'Options of QA : {option_txt}\nType : {sample['question_type']}\nDuration (sec) : {sample['duration']}'
                Total_QA[db][sample['question_id']] = {
                    "question" : sample['question'],
                    "video" : sample['video'],
                    "options" : sample['candidates'],
                    "answer" : answer,
                    "answer_text" : sample['answer'],
                    "meta" : meta_info,
                }


        elif db == 'MVBench':
            for sample in data:
                if sample['video'] in buffer:
                    while True:
                        sample['video'] = sample['video'] + '@'
                        if sample['video'] not in buffer:
                            buffer.append(sample['video'])
                            break
                else:
                    buffer.append(sample['video'])

                option_txt = ''
                for idx, opt in enumerate(sample['candidates']):
                    option_txt += chr(idx + 65) + '. ' + opt + '. '
                    sample['candidates'][idx] = chr(idx + 65) + '. ' + opt
                    if opt == sample['answer']:
                        answer = chr(idx + 65)
                meta_info = f'Options of QA : {option_txt}'
                Total_QA[db][sample['video']] = {
                    "question" : sample['question'],
                    "video" : sample['video'].replace('@',''),
                    "options" : sample['candidates'],
                    "answer" : answer,
                    "answer_text" : sample['answer'],
                    "meta" : meta_info,
                }

        elif db == 'RTV-Bench':
            for sample in data:
                option_txt = ''
                options = []
                for opt in sample['options'].keys():
                    option_txt += opt + '. ' + sample['options'][opt] + '. '
                    options.append(opt + '. ' + sample['options'][opt])
                    if opt == sample['answer']:
                        answer_txt = sample['options'][opt]
                meta_info = f'Options of QA : {option_txt}\nType : {sample['type']}\nField : {sample['field']}'
                Total_QA[db][sample['questionID']] = {
                    "question" : sample['question'],
                    "video" : sample['video'],
                    "options" : options,
                    "answer" : sample['answer'],
                    "answer_text" : answer_txt,
                    "meta" : meta_info,
                }

        elif db == 'TVBench':
            for sample in data:
                if sample['video'] in buffer:
                    while True:
                        sample['video'] = sample['video'] + '@'
                        if sample['video'] not in buffer:
                            buffer.append(sample['video'])
                            break
                else:
                    buffer.append(sample['video'])
                option_txt = ''
                for idx, opt in enumerate(sample['candidates']):
                    option_txt += chr(idx + 65) + '. ' + opt + ' '
                    sample['candidates'][idx] = chr(idx + 65) + '. ' + opt
                    if opt == sample['answer']:
                        answer = chr(idx + 65)
                import pdb;pdb.set_trace()

                meta_info = f'Options of QA : {option_txt}'
                Total_QA[db][sample['video']] = {
                    "question" : sample['question'],
                    "video" : sample['video'].replace('@',''),
                    "options" : sample['candidates'],
                    "answer" : answer,
                    "answer_text" : sample['answer'],
                    "meta" : meta_info,
                }


        elif db == 'VCR-Bench':
            for idx in range(len(data)):
                if data.iloc[idx]['answer'].upper() in ['A','B','C','D','E','F','G','H']:
                    question = data.iloc[idx]['question'].split('\n')[0]
                    options = data.iloc[idx]['question'].split('\n')[1:-1]
                    option_txt = ''
                    for opt in options:
                        option_txt += opt + ' '
                        if opt[0] == data.iloc[idx]['answer']:
                            answer_txt = opt[3:]
                    meta_info = f'Options of QA : {option_txt}\nType : {data.iloc[idx]['dimension']}\nReasoning : {data.iloc[idx]['reasoning']} \nAnswer : {answer_txt}'
                    Total_QA[db][data.iloc[idx]['id'].item()] = {
                        "question" : question,
                        "video" : data.iloc[idx]['video_path'],
                        "options" : options,
                        "answer" : data.iloc[idx]['answer'],
                        "answer_text" : answer_txt,
                        "meta" : meta_info,
                    }

        elif db == 'VSI-Bench':
            for sample in data:
                if sample['ground_truth'] in ['A','B','C','D','E','F','G','H']:
                    option_txt = ''
                    for opt in sample['options']:
                        option_txt += opt + '. '
                        if opt[0] == sample['ground_truth']:
                            answer_txt = opt[3:]
                    meta_info = f'Options of QA : {option_txt}\nCategory : {sample['question_type']}'
                    Total_QA[db][sample['id']] = {
                        "question" : sample['question'],
                        "video" : sample['dataset'] + '_' + sample['scene_name'],
                        "options" : sample['options'],
                        "answer" : sample['ground_truth'],
                        "answer_text" : answer_txt,
                        "meta" : meta_info,
                    }

        elif db == 'Video-Holmes':
            type_dict = {
                "SR" : 'Social Reasoning (Inferring social relationships between characters. This includes identifying identity associations across time (e.g., the same man in youth and old age))',
                "PAR" : 'Physical Anomaly Reasoning (Identifying scenes in the video that deviate from real-world norms and reasoning about their underlying rules or implicit meanings)',
                "MHR" : 'Multimodal Hint Reasoning (Decoding cues or fact from multimodal hints, such as semantic implications of camera movements or gradual changes in object positions)',
                "IMC" : 'Intention & Motive Chaining (Observing characters’ actions or environmental cues to disentangle surface behaviors from underlying behavioral intentions)',
                "TCI" : 'Temporal Causal Inference (Inferring causal mechanisms between events across time and space using cinematic language and multimodal clues)',
                "TA" : 'Timeline Analysis (Integrating and reconstructing the narrative storyline of the film)',
                "CTI" : 'Core Theme Inference (Extracting the core theme or deeper meaning of the video by analyzing its plot, dialogues, and symbolic elements)',
            }
            for sample in data:
                option_txt = ''
                options = []
                for opt in sample['Options'].keys():
                    option_txt += opt + '. ' + sample['Options'][opt] + '. '
                    options.append(opt + '. ' + sample['Options'][opt])
                    if opt == sample['Answer']:
                        answer_txt = sample['Options'][opt]
                meta_info = f'Options of QA : {option_txt}\nType of QA :{type_dict[sample['Question Type']]}\nReasoning : {sample['Explanation']}\nAnswer : {answer_txt}'
                Total_QA[db][sample['Question ID']] = {
                    "question" : sample['Question'],
                    "video" : sample['video ID'],
                    "options" : options,
                    "answer" : sample['Answer'],
                    "answer_text" : answer_txt,
                    "meta" : meta_info,
                }

        elif db == 'Video-MME':
            for idx in range(len(data)):
                option_txt = ''
                for opt in data.iloc[idx]['options']:
                    option_txt += opt + ' '
                    if opt[0] == data.iloc[idx]['answer']:
                        answer_txt = opt[3:]
                meta_info = f'Options of QA : {option_txt}\nDomain : {data.iloc[idx]['domain']}\nDuration : {data.iloc[idx]['duration']}\nCategory : {data.iloc[idx]['sub_category']}'
                Total_QA[db][data.iloc[idx]['question_id']] = {
                    "question" : data.iloc[idx]['question'],
                    "video" : data.iloc[idx]['videoID'],
                    "options" : data.iloc[idx]['options'].tolist(),
                    "answer" : data.iloc[idx]['answer'],
                    "answer_text" : answer_txt,
                    "meta" : meta_info,
                }

        elif db == 'egoschema':
            for idx in range(len(data)):
                option_txt = ''
                for opt in data.iloc[idx]['option']:
                    option_txt += opt + ' '
                    if opt[0] == chr(int(data.iloc[idx]['answer']) + 65):
                        answer_txt = opt[3:]
                meta_info = f'Options of QA : {option_txt}'
                Total_QA[db][data.iloc[idx]['question_idx']] = {
                    "question" : data.iloc[idx]['question'],
                    "video" : data.iloc[idx]['video_idx'],
                    "options" : data.iloc[idx]['option'].tolist(),
                    "answer" : chr(int(data.iloc[idx]['answer']) + 65),
                    "answer_text" : answer_txt,
                    "meta" : meta_info,
                }

        elif db == 'MINERVA':
            for sample in data:
                option_txt = ''
                options = []
                for idx in [0,1,2,3,4]:
                    option_txt += chr(idx + 65) + '. ' + sample[f'answer_choice_{idx}'] + ' '
                    options.append(chr(idx + 65) + '. ' + sample[f'answer_choice_{idx}'])
                meta_info = f'Options of QA : {option_txt}\nType of QA : {sample['question_type']}\nReasoning : {sample['reasoning']}\nAnswer : {sample['answer']}'
                Total_QA[db][sample['key']] = {
                    "question" : sample['question'],
                    "video" : sample['video_id'],
                    "options" : options,
                    "answer" : chr(sample['answer_id'] + 65),
                    "answer_text" : sample['answer'],
                    "meta" : meta_info,
                }

        elif db == 'MMR-V':
            for idx in range(len(data)):
                option_txt = ''
                options = []
                for _idx, opt in enumerate(data.iloc[idx]['options']):
                    opt = opt.split(')')[-1]
                    opt = opt[1:]
                    option_txt += chr(_idx + 65) + '. ' + opt + ' '
                    options.append(chr(_idx + 65) + '. ' + opt)
                    if chr(_idx + 65) == data.iloc[idx]['correctAnswer'][1]:
                        answer_txt = opt
                meta_info = f'Options of QA : {option_txt}'
                Total_QA[db][str(data['question_idx'][idx])] = {
                    "question" : data.iloc[idx]['question'],
                    "video" : data.iloc[idx]['video'],
                    "options" : options,
                    "answer" : data.iloc[idx]['correctAnswer'][1],
                    "answer_text" : answer_txt,
                    "meta" : meta_info,
                    "abilityType_L2" : data.iloc[idx]['video'],
                    "abilityType_L3" : data.iloc[idx]['video'],

                }

    total_cnt = 0
    for db in Total_QA.keys():
        print("DB : {:25s} || Num of QA : {:5d}".format(db,len(Total_QA[db])))
        total_cnt += len(Total_QA[db])
    print(f"# of Total QA : {total_cnt}")

    with open("./video_total.json", "w") as json_file:
        json.dump(Total_QA, json_file, indent=4, cls=NpEncoder)
