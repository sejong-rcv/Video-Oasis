import json
import os
import tqdm

from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress

def download_youtube_video(url, vid, save_path='./'):
    try:
        yt = YouTube(url, on_progress_callback=on_progress, use_oauth=True)
        stream = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            print(f"[INFO] Downloading: {stream.resolution}")
            stream.download(output_path=save_path, filename=f"{vid}.mp4")
            print(f"[SUCCESS] Download complete: {yt.title}")
        else:
            print("[ERROR] Could not find a suitable stream.")
    except Exception as e:
        print(f"[EXCEPTION] Error occurred: {e}")

if __name__ == '__main__':
    os.makedirs("./videos",exist_ok=True)

    anno_list = json.load(open("../../annos/minerva/minerva.json"))
    url_list = set()
    for ann in tqdm.tqdm(anno_list):
        url = 'https://www.youtube.com/watch?v='+ann['video_id']
        url_list.add(url)

    for url in tqdm.tqdm(url_list):
        vid = url.split('v=')[-1]
        download_youtube_video(url, vid, "./videos")