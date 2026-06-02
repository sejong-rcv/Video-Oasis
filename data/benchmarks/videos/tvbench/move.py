import os
import shutil
from tqdm import tqdm  # 진행률 표시 (없으면 pip install tqdm 필요, 혹은 아래 설명 참조)

# ================= 설정 구간 =================
# 1. 원본 경로 (검사할 폴더)
SOURCE_DIR = "/mnt/users/gtlim/workspace/src/benchmark/videos/tvbench/videos"

# 2. 목적지 경로 (여기로 파일들이 복사됩니다) -> 원하시는 경로로 수정하세요!
TARGET_DIR = "/mnt/users/gtlim/workspace/src/benchmark/videos/tvbench/total_videos"

# 3. 복사할 파일 확장자 (비디오만 옮기고 싶을 때)
# 모든 파일을 다 옮기려면 이 리스트를 비우거나 검사 로직을 수정하면 됩니다.
# ============================================

def copy_files():
    # 목적지 폴더가 없으면 생성
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📁 폴더 생성 완료: {TARGET_DIR}")

    files_to_copy = []

    # 1. 파일 목록 스캔
    print(f"🔍 '{SOURCE_DIR}' 경로 스캔 중...")
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            src_path = os.path.join(root, file)
            files_to_copy.append(src_path)

    total_files = len(files_to_copy)
    print(f"✅ 총 {total_files}개의 대상 파일을 찾았습니다.")

    if total_files == 0:
        print("이동할 파일이 없습니다.")
        return

    # 2. 파일 복사 실행
    print("🚀 복사 시작...")
    
    # tqdm이 설치되어 있다면 진행바 표시, 아니면 일반 for문 사용
    # 설치가 안 되어 있다면: pip install tqdm
    for src_path in tqdm(files_to_copy, desc="Copying", unit="file"):
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(TARGET_DIR, file_name)

        # 중복 이름 처리 (이미 같은 파일이 있으면 건너뛰거나 이름 변경 로직 추가 가능)
        if os.path.exists(dst_path):
            print(f"⚠️ 중복 파일 건너뜀: {file_name}")
            continue

        try:
            shutil.copy2(src_path, dst_path) # copy2는 메타데이터 유지
        except Exception as e:
            print(f"❌ 에러 발생 ({file_name}): {e}")

    print("\n🎉 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    copy_files()