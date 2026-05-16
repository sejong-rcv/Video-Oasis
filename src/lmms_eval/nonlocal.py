import json
import os
total_ann = json.load(open("vqa_total.json"))
for ann in total_ann:
    ann['video_path'] = '/mnt/gtlim_data/users/gtlim/benchmark' + ann['video_path']
with open(os.path.join("vqa_total_nonlocal.json"), "w") as json_file:
    json.dump(total_ann, json_file, indent=4)