import os
from tqdm import tqdm 

if __name__ == '__main__':

    files_to_copy = []
    ext = set()
    SOURCE_DIR = '/mnt/users/gtlim/workspace/src/benchmark/videos/tvbench/videos'
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            src_path = os.path.join(root, file)
            ext.add(src_path.split('.')[-1])
    import pdb;pdb.set_trace()