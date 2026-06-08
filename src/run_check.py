import os
import json
if __name__ == '__main__':
    total_ann = json.load(open("./lmms_eval/video_total.json"))
    db_dict = dict()
    for ann in total_ann:
        if ann['db'] not in db_dict.keys():
            db_dict[ann['db']]=[]

        if os.path.isfile(ann['video_path'])==False:
            db_dict[ann['db']].append(ann['video_path'])

    for db in db_dict.keys():
        print("DB : {:25s} || Missing : {}".format(db, len(db_dict[db])))

    import pdb;pdb.set_trace()