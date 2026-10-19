import os
import numpy as np
import subprocess
import tqdm
from decord import VideoReader, cpu
from multiprocessing import Pool, cpu_count

def is_video_healthy(video_path, num_frames=16):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
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
            '-map', '0:v:0', '-map', '0:a?',
            '-c:v', 'libx264', '-crf', '23',
            '-preset', 'fast',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            '-threads', '2',
            temp_output_path
        ]
    elif input_path.lower().endswith('.webm'):
        command = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', input_path,
            '-map', '0:v:0', '-map', '0:a?',
            '-c:v', 'libvpx-vp9', '-b:v', '0',
            '-crf', '30', '-preset', 'fast',
            '-c:a', 'copy',
            '-threads', '2',
            temp_output_path
        ]
    else:
        return input_path, False, f"unsupported extension: {ext}"

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        
        if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
            if is_video_healthy(temp_output_path):
                os.replace(temp_output_path, input_path)
                return input_path, True, ""
            else:
                os.remove(temp_output_path)
                return input_path, False, "converted file failed the decord health check"
        return input_path, False, "ffmpeg did not create a non-empty output file"
        
    except Exception as e:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        if isinstance(e, subprocess.CalledProcessError):
            message = (e.stderr or str(e)).strip()
        else:
            message = str(e)
        return input_path, False, message

def main():

    if os.path.isfile('failed_list.txt'):
        failed_video_list = open('failed_list.txt', encoding='utf-8').read().splitlines()
        failed_video_list = check_corrupted_videos(failed_video_list, num_frames=16)
    else:
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
        print(*failed_video_list, sep='\n', file=open('failed_list.txt', 'w', encoding='utf-8'))

    if not failed_video_list:
        return
            
    num_processes = min(8, cpu_count() or 1)
    print(f"Re-encoding {len(failed_video_list)} videos with {num_processes} workers")
    with open('ffmpeg_errors.log', 'w', encoding='utf-8') as error_log:
        with Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(fix_video_ffmpeg, failed_video_list)
            for video_path, succeeded, message in tqdm.tqdm(results, total=len(failed_video_list), unit="vid"):
                if not succeeded:
                    failure = f"[FAILED] {video_path}: {message}"
                    tqdm.tqdm.write(failure)
                    print(failure, file=error_log, flush=True)

    remaining_failed = check_corrupted_videos(failed_video_list, num_frames=16)
    with open('failed_list.txt', 'w', encoding='utf-8') as f:
        print(*remaining_failed, sep='\n', file=f)
        
if __name__ == "__main__":
    main()
