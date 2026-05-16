import json
import os
import tqdm

from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress


def download_youtube_video(url, save_path='./'):
    """
    YouTube 영상 ID를 받아 영상을 다운로드합니다.

    Parameters:
        vid (str): YouTube 영상 ID (예: 'dQw4w9WgXcQ')
        save_path (str): 영상을 저장할 경로 (기본: 현재 디렉토리)
    """
    try:
        yt = YouTube(url, on_progress_callback = on_progress)

        # 가장 높은 해상도의 progressive 스트림 선택 (영상+오디오 포함)
        # stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        stream = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()

        if stream:
            print(f"[INFO] 다운로드 중: {stream.resolution}")
            stream.download(output_path=save_path)
            print(f"[SUCCESS] 다운로드 완료: {yt.title}")
        else:
            print("[ERROR] 적절한 스트림을 찾을 수 없습니다.")
    except Exception as e:
        print(f"[EXCEPTION] 에러 발생: {e}")

if __name__ == '__main__':
    with open("video_info.meta.jsonl") as f:
        anno_list = [json.loads(line) for line in f]

    url_list = set()

    for ann in tqdm.tqdm(anno_list):
        url = 'https://www.youtube.com/watch?v=' + ann['key']
        url_list.add(url)

    cnt = 0

    for url in tqdm.tqdm(url_list):
        vid = url.split('v=')[-1]
        save_path = os.path.join("./videos", vid)
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        file_list = os.listdir(save_path)
        if len(file_list) == 0:
            cnt += 1 
            print(url)
            download_youtube_video(url, save_path)

    # import pdb;pdb.set_trace()