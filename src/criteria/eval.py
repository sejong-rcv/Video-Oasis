
import os
import json
import tqdm
import numpy as np
import pandas as pd

from tabulate import tabulate

def get_db_wise_dict(preds):
    output = dict()
    for pred in preds:
        if pred['db'] not in output.keys():
            output[pred['db']] = dict()
            output[pred['db']]['correct_cnt'] = 0
            output[pred['db']]['valid_cnt'] = 0
            output[pred['db']]['rand_acc'] = []

    output['Total'] = dict()
    output['Total']['correct_cnt'] = 0
    output['Total']['valid_cnt'] = 0
    output['Total']['rand_acc'] = []

    return output

if __name__ == '__main__':

    total_db = dict()
    total_results = dict()
    vocab = dict()

    vocab[0] = []
    vocab[1] = []
    vocab[2] = []
    vocab[3] = []

    rand_acc = []
    candidates = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

    for file in tqdm.tqdm(os.listdir("/mnt/users/gtlim/workspace/src/criteria/bias/json")):
        model = file.replace('.json','')
        if model not in ['llama', 'mistral', 'qwen']:
            continue
        preds = json.load(open(f"/mnt/users/gtlim/workspace/src/criteria/bias/json/{file}"))
        total_results[model] = get_db_wise_dict(preds)
        for pred in tqdm.tqdm(preds):
            if len(pred['options'])==0:
                continue
            else:
                total_results[model][pred['db']]['rand_acc'].append(1/len(pred['options']))
                total_results[model]['Total']['rand_acc'].append(1/len(pred['options']))

                key = pred['key']
                if key not in total_db.keys():
                    total_db[key] = 0

                if type(pred['pred']) == dict:
                    if 'content' not in pred['pred'].keys():
                        import pdb;pdb.set_trace()
                    else:
                        pred['pred'] = pred['pred']['content']

                pred['pred']=pred['pred'].split('boxed{')[-1][0]

                if pred['pred'] not in candidates:
                    continue
                else:
                    if pred['pred'] not in candidates[:len(pred['options'])]:
                        continue
                    else:
                        total_results[model][pred['db']]['valid_cnt'] += 1
                        total_results[model]['Total']['valid_cnt'] += 1
                        if pred['pred'] == pred['answer']:
                            total_results[model][pred['db']]['correct_cnt'] += 1
                            total_results[model]['Total']['correct_cnt'] += 1
                            total_db[key] += 1

    table_data = []
    for model in ['random','llama', 'mistral', 'qwen']:
        row = [model] 
        for db in ['egoschema', 'ImplicitQA', 'MINERVA', 'RTV-Bench', 'VSI-Bench', 'MVBench', 'LongVideoBench', 'LVBench', 'MLVU_Test', 'MMR-V', 'TVBench','VCR-Bench', 'Video-Holmes', 'Video-MME', 'Total']:
            if model == 'random':
                acc = 100 * np.mean(total_results['llama'][db]['rand_acc']).item()
                row.append(f"{acc:.2f}") 
            else:
                acc = 100 * total_results[model][db]['correct_cnt'] / total_results[model][db]['valid_cnt']
                row.append(f"{acc:.2f}") 
        table_data.append(row)
    headers = ["Model"] + ['egoschema', 'ImplicitQA', 'MINERVA', 'RTV-Bench', 'VSI-Bench', 'MVBench', 'LongVideoBench', 'LVBench', 'MLVU_Test', 'MMR-V', 'TVBench','VCR-Bench', 'Video-Holmes', 'Video-MME', 'Total']
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    csv_data = {}
    for model in ['random','llama', 'mistral', 'qwen']:
        csv_data[model] = {}
        for db in ['egoschema', 'ImplicitQA', 'MINERVA', 'RTV-Bench', 'VSI-Bench', 'MVBench', 'LongVideoBench', 'LVBench', 'MLVU_Test', 'MMR-V', 'TVBench','VCR-Bench', 'Video-Holmes', 'Video-MME', 'Total']:
            if model == 'random':
                acc = 100 * np.mean(total_results['llama'][db]['rand_acc']).item()
            else:
                acc = 100 * total_results[model][db]['correct_cnt'] / total_results[model][db]['valid_cnt']
            csv_data[model][db] = acc
    df = pd.DataFrame(csv_data).T
    df.to_csv("./bias_llm_results.csv")

    for key in total_db.keys():
        vocab[total_db[key]].append(key)

    for i in range(1,len(vocab.keys())):
        for j in range(i+1,len(vocab.keys())):
            vocab[i].extend(vocab[j])

    for key in vocab.keys():
        print("Key : {} and Cnt : {}".format(key, len(vocab[key])))

    with open(f"./bias_llm_vocab.json", "w") as json_file:
        json.dump(vocab, json_file, indent=4)