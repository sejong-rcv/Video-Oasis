
import os
import json
import tqdm
import argparse
import numpy as np
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Eval Language Only")
    parser.add_argument('--model', type=str, choices=['eagle25','qwen3_vl','qwen25_vl','llama','mistral','qwen'])
    args = parser.parse_args()
    model = args.model
    preds = []
    for file in tqdm.tqdm(os.listdir(os.path.join('output',model))):
        key = file.replace('.json','')
        pred = json.load(open(os.path.join('output',model,file)))
        pred['key'] = key
        pred['db'] = key.split('**@@**')[0]
        pred['qid'] = key.split('**@@**')[1]
        preds.append(pred)
    with open(f"./json/{model}.json", "w") as json_file:
        json.dump(preds, json_file, indent=4)
