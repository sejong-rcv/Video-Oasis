import os
import shutil
from tqdm import tqdm  # 진행률 표시 (없으면 pip install tqdm)

# ================= 설정 =================
# 현재 위치(.)에 있는 폴더들을 스캔합니다.
SOURCE_DIR = "/mnt/users/gtlim/workspace/src/benchmark/videos/minerva/videos_"  
# 파일들을 모을 경로 (현재 위치의 상위 폴더에 'collected_videos' 생성)
TARGET_DIR = "/mnt/users/gtlim/workspace/src/benchmark/videos/minerva/videos" 
# 이동(move)할지 복사(copy)할지 선택 (True면 원본 삭제됨)
MOVE_FILES = False 
# =======================================

def process_videos():
    # 타겟 디렉토리가 없으면 생성
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 현재 경로의 모든 항목 중 디렉토리만 리스트로 가져오기
    folders = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"📂 총 {len(folders)}개의 비디오 폴더를 찾았습니다. 작업을 시작합니다...")

    # 진행률바(tqdm)와 함께 반복
    for folder_name in tqdm(folders):
        folder_path = os.path.join(SOURCE_DIR, folder_name)
        
        # 폴더 내의 파일 리스트 확인
        files = os.listdir(folder_path)
        
        if not files:
            continue # 빈 폴더면 건너뜀

        # 폴더 안의 첫 번째 파일 선택 (보통 파일이 하나라고 하셨으므로)
        original_filename = files[0]
        original_file_path = os.path.join(folder_path, original_filename)

        # 원본 파일의 확장자 가져오기 (예: .mp4, .mkv 등)
        _, ext = os.path.splitext(original_filename)
        
        # 확장자가 없으면 .mp4로 가정 (필요시 주석 해제)
        # if not ext: ext = ".mp4"

        # 새로운 파일명: 폴더명(vid) + 원본확장자
        new_filename = f"{folder_name}{ext}"
        target_file_path = os.path.join(TARGET_DIR, new_filename)

        try:
            if MOVE_FILES:
                shutil.move(original_file_path, target_file_path)
            else:
                shutil.copy2(original_file_path, target_file_path) # 메타데이터 유지 복사
        except Exception as e:
            print(f"\n[Error] {folder_name} 처리 중 오류 발생: {e}")

    action = "이동" if MOVE_FILES else "복사"
    print(f"\n✅ 모든 작업이 완료되었습니다! ({action} 완료)")
    print(f"📁 저장 위치: {os.path.abspath(TARGET_DIR)}")

if __name__ == "__main__":
    process_videos()