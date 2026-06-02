import os
import multiprocessing as mp
from tqdm import tqdm
from modules.data_loader import load_video_mme_data
from modules.audio_utils import AudioProcessor

# 설정
JSON_PATH = "/home/jylee/workspace/Whisper/datasets/videomme_all.json"

def main():
    # 1. 데이터 로드
    try:
        data = load_video_mme_data(JSON_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 중복 비디오 경로 제거 (하나의 비디오에 여러 질문이 있는 경우 대비)
    unique_videos = list({item['video_path']: item for item in data}.values())
    video_paths = [v['video_path'] for v in unique_videos]
    total_videos = len(video_paths)

    # 2. 오디오 프로세서 초기화
    processor = AudioProcessor()

    print(f"\n[Phase 1] Audio Extraction Started")
    print(f" - Total Unique Videos: {total_videos}")
    print(f" - Output Directory: {processor.temp_dir}")
    print(f" - Workers: 16\n")

    # 3. 병렬 처리 및 결과 수집
    # processor.extract_audio는 성공 시 경로를, 실패 시 None을 반환합니다.
    with mp.Pool(16) as pool:
        results = list(tqdm(pool.imap(processor.extract_audio, video_paths), 
                           total=total_videos, 
                           desc="Extracting Audio"))

    # 4. 통계 계산
    success_cnt = sum(1 for r in results if r is not None)
    fail_cnt = total_videos - success_cnt
    success_rate = (success_cnt / total_videos * 100) if total_videos > 0 else 0

    # 5. 최종 결과 출력
    print("\n" + "="*40)
    print(f"  Extraction Summary")
    print("="*40)
    print(f"  - Total Videos   : {total_videos}")
    print(f"  - Success        : {success_cnt}")
    print(f"  - Failed         : {fail_cnt}")
    print(f"  - Success Rate   : {success_rate:.2f}%")
    print("="*40)

    if fail_cnt > 0:
        print(f" * Check your video files if 'Failed' count is high.")
    print(f"[Done] Audio extraction process completed.\n")

if __name__ == "__main__":
    main()