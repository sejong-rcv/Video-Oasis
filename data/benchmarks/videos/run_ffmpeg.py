import os
import numpy as np
import subprocess
import tqdm
from decord import VideoReader, cpu
from multiprocessing import Pool, cpu_count

def is_video_healthy(video_path, num_frames=16):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return False
        
        extract_count = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, extract_count).astype(int)
        batch = vr.get_batch(indices)
        
        if batch.shape[0] == 0:
            return False
        return True
    except Exception:
        return False

def check_corrupted_videos(video_list, num_frames=16):
    failed_videos = []
    for video_path in tqdm.tqdm(video_list, desc="Checking Videos"):
        if not is_video_healthy(video_path, num_frames):
            failed_videos.append(video_path)
    return failed_videos

def fix_video_ffmpeg(args):
    input_path = args
    directory, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    
    temp_output_path = os.path.join(directory, f"{name}_temp{ext}")

    if input_path.lower().endswith('.mp4'):
        command = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', input_path,
            '-c:v', 'libx264', '-crf', '23',
            '-preset', 'fast', '-r', '30',
            '-movflags', '+faststart',
            '-threads', '2',
            temp_output_path
        ]
    elif input_path.lower().endswith('.webm'):
        command = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', input_path,
            '-c:v', 'libvpx-vp9', '-b:v', '0',
            '-crf', '30', '-preset', 'fast', '-r', '30',
            '-threads', '2',
            temp_output_path
        ]
    else:
        return False

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=600)
        
        if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
            if is_video_healthy(temp_output_path):
                os.replace(temp_output_path, input_path)
                return True
            else:
                os.remove(temp_output_path)
                return False
        return False
        
    except Exception as e:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return False

def main():
    root_dir = './'
    video_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not file.endswith('.py'):
                full_path = os.path.join(dirpath, file)
                _, ext = os.path.splitext(full_path)
                if ext.lower() in ['.mp4', '.webm']:
                    video_list.append(full_path)

    failed_video_list = check_corrupted_videos(video_list, num_frames=16)

    if not failed_video_list:
        return
            
    num_processes = max(1, int(cpu_count() * 0.4)) 
    with Pool(processes=num_processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(fix_video_ffmpeg, failed_video_list), total=len(failed_video_list), unit="vid"))
        
if __name__ == "__main__":
    main()