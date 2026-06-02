import os
import subprocess
import json
import tqdm
from multiprocessing import Pool, cpu_count

def fix_video_ffmpeg(args):

    input_path, output_path = args
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    if input_path.endswith('.mp4'):
        command = [
            'ffmpeg',
            '-y',
            '-v', 'error',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'fast',    
            '-r', '30',
            '-movflags', '+faststart',
            '-threads', '2',       
            output_path
        ]

    if input_path.endswith('.webm'):
        command = [
            'ffmpeg',
            '-y',
            '-v', 'error',
            '-i', input_path,
            '-c:v', 'libvpx-vp9',
            '-b:v', '0',
            '-crf', '30',
            '-preset', 'fast',    
            '-r', '30',
            '-threads', '2',       
            output_path
        ]

    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(e)
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def main():
    json_path = "decord_error_list.json"
    if not os.path.exists(json_path):
        return

    error_dict = json.load(open(json_path))
    
    tasks = []
    
    for db in error_dict.keys():
        save_dir = os.path.join("./new_videos", db)
        os.makedirs(save_dir, exist_ok=True)
        
        if len(error_dict[db]) != 0:
            for vid_path in error_dict[db]:
                vid_name = vid_path.split('/')[-1]
                new_vid_path = os.path.join(save_dir, vid_name)
                
                tasks.append((vid_path, new_vid_path))

    num_processes = max(1, int(cpu_count() * 0.4)) 
    
    print(f"Multiprocessing (Processes: {num_processes})")
    
    with Pool(processes=num_processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(fix_video_ffmpeg, tasks), total=len(tasks), unit="vid"))

if __name__ == "__main__":
    main()