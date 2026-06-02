import pandas as pd
import json
import os
import tqdm
if __name__ == '__main__':
    data = pd.read_parquet('./videomme/test-00000-of-00001.parquet', engine='pyarrow')
    Total_QA = list()
    for idx in range(len(data)):
        option_txt = ''
        for opt in data.iloc[idx]['options']:
            option_txt += opt + ' '
            if opt[0] == data.iloc[idx]['answer']:
                answer_txt = opt[3:]
        meta_info = f'Options of QA : {option_txt}\nDomain : {data.iloc[idx]['domain']}\nDuration : {data.iloc[idx]['duration']}\nCategory : {data.iloc[idx]['sub_category']}'
        vid_path = os.path.join('/mnt/users/gtlim/workspace/data/video-mme/videos',data.iloc[idx]['videoID']+'.mp4')
        if os.path.isfile(vid_path)==True:
            item = {
                "db" : "videomme",
                "question" : data.iloc[idx]['question'],
                "video" : vid_path,
                "qid" : data.iloc[idx]['question_id'],
                "options" : data.iloc[idx]['options'].tolist(),
                "answer" : data.iloc[idx]['answer'],
                "answer_text" : answer_txt,
                "meta" : meta_info,
                "duration" : data.iloc[idx]['duration'],
                "task_type" : data.iloc[idx]['task_type']
            }
            Total_QA.append(item)
        else:
            import pdb;pdb.set_trace()
    with open("videomme.json", "w") as json_file:
        json.dump(Total_QA, json_file, indent=4)