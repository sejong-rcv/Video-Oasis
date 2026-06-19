# import os
# import json
# import re
# import tqdm
# import shutil

# # ==========================================
# # ⚙️ 설정 (Configuration)
# # ==========================================
# # 원본 STT 폴더 경로
# SRC_ROOT = '/data3/jylee/workspace/Whisper/STT'

# # 깨끗한 파일을 저장할 새로운 폴더 경로
# DST_ROOT = '/data3/jylee/workspace/Whisper/STT_clean'

# # ==========================================
# # 🛡️ 필터링 로직 (Filtering Logic)
# # ==========================================
# def is_valid_transcript(stt_data):
#     """
#     STT 데이터 유효성 검사 (True: 정상 / False: 노이즈)
#     """
#     if not stt_data or "transcript" not in stt_data:
#         return False

#     full_text = stt_data.get("transcript", "").strip()
    
#     # 1. 길이 검사 (너무 짧으면 삭제)
#     if len(full_text) < 4:
#         return False

#     # 2. 환각 키워드 검사
#     HALLUCINATION_KEYWORDS = [
#         "시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", 
#         "구독과 좋아요", "Subscribe", "Thank you for watch", 
#         "MBC News", "MBC 뉴스", "SBS 뉴스", "YTN", "Copyright", 
#         "자막 제작", "Subtitles by", "Amara.org", 
#         "배경 음악", "[Music]", "(Music)", "음악", 
#         "박수", "(Applause)", "[Applause]"
#     ]
#     for keyword in HALLUCINATION_KEYWORDS:
#         if keyword.lower() in full_text.lower():
#             return False

#     # 3. 허용된 언어/문자 비율 검사 (외계어, 비정상 문자 차단)
#     # 영어, 한국어, 숫자, 기본 특수문자만 허용
#     allowed_pattern = re.compile(r"[a-zA-Z0-9가-힣\s\.,\?!'\"\(\)\[\]-]")
#     allowed_char_count = len(allowed_pattern.findall(full_text))
#     total_char_count = len(full_text)
    
#     if total_char_count > 0:
#         valid_ratio = allowed_char_count / total_char_count
#         # 유효 문자가 50% 미만이면 노이즈 (예: ༱༱༱, 이모지 폭탄 등)
#         if valid_ratio < 0.5: 
#             return False

#     # 4. 반복 패턴 검사 (같은 단어 무한 반복)
#     words = full_text.split()
#     if len(words) > 5:
#         unique_words = set(words)
#         ratio = len(unique_words) / len(words)
#         # 고유 단어 비율이 20% 미만이면 삭제
#         if ratio < 0.2: 
#             return False

#     return True

# # ==========================================
# # 🚀 메인 처리 (Processing)
# # ==========================================
# def main():
#     if not os.path.exists(SRC_ROOT):
#         print(f"❌ Error: Source directory '{SRC_ROOT}' not found.")
#         return

#     # STT 폴더 내의 모든 데이터셋 폴더 탐색 (video-mme, mvbench 등)
#     dataset_dirs = [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))]
    
#     print(f"📂 Found Datasets: {dataset_dirs}")
#     print(f"🚀 Preprocessing STT files from '{SRC_ROOT}' to '{DST_ROOT}'...\n")

#     total_files = 0
#     kept_files = 0
#     removed_files = 0

#     for dataset in dataset_dirs:
#         src_dataset_path = os.path.join(SRC_ROOT, dataset)
#         dst_dataset_path = os.path.join(DST_ROOT, dataset)

#         # 타겟 폴더 생성
#         os.makedirs(dst_dataset_path, exist_ok=True)

#         files = [f for f in os.listdir(src_dataset_path) if f.endswith(".json")]
        
#         print(f"   Processing '{dataset}' ({len(files)} files)...")

#         for filename in tqdm.tqdm(files, desc=f"   Filtering {dataset}", leave=False):
#             total_files += 1
#             src_file = os.path.join(src_dataset_path, filename)
#             dst_file = os.path.join(dst_dataset_path, filename)

#             try:
#                 with open(src_file, 'r', encoding='utf-8') as f:
#                     data = json.load(f)

#                 # 🔥 필터 통과하면 저장, 아니면 버림
#                 if is_valid_transcript(data):
#                     # 그대로 복사 (또는 다시 덤프)
#                     with open(dst_file, 'w', encoding='utf-8') as f:
#                         json.dump(data, f, indent=4, ensure_ascii=False)
#                     kept_files += 1
#                 else:
#                     # 노이즈 파일은 저장하지 않음 (건너뜀)
#                     removed_files += 1

#             except Exception as e:
#                 print(f"⚠️ Error reading {filename}: {e}")

#     print("\n" + "="*50)
#     print("✅ Preprocessing Completed!")
#     print(f"📊 Total Processed : {total_files}")
#     print(f"🟢 Kept (Valid)    : {kept_files}")
#     print(f"🔴 Removed (Noise) : {removed_files}")
#     print(f"📁 Clean Data Path : {DST_ROOT}")
#     print("="*50)

# if __name__ == "__main__":
#     main()

import os
import json
import re
import tqdm
import shutil

# ==========================================
# ⚙️ 설정 (Configuration)
# ==========================================
# 원본 STT 폴더 경로
SRC_ROOT = '/data3/jylee/workspace/Whisper/STT'

# 깨끗한 파일을 저장할 폴더 (Valid)
DST_ROOT = '/data3/jylee/workspace/Whisper/STT_clean_2'

# 🗑️ 걸러진(노이즈) 파일을 저장할 폴더 (Filtered)
FILTERED_ROOT = '/data3/jylee/workspace/Whisper/STT_filtered'

# ==========================================
# 🛡️ 필터링 로직 (Filtering Logic)
# ==========================================
def is_valid_transcript(stt_data):
    """
    STT 데이터 유효성 검사 (True: 정상 / False: 노이즈)
    """
    if not stt_data or "transcript" not in stt_data:
        return False

    full_text = stt_data.get("transcript", "").strip()
    
    # 1. 길이 검사 (너무 짧으면 삭제)
    if len(full_text) < 4:
        return False

    # 2. 환각 키워드 검사
    HALLUCINATION_KEYWORDS = [
        "시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", 
        "구독과 좋아요", 
        "YTN", "Copyright", 
        "자막 제작", "Subtitles by", "Amara.org", 
        "배경 음악", "[Music]", "(Music)", "음악", 
        "박수", "(Applause)", "[Applause]"
    ]
    for keyword in HALLUCINATION_KEYWORDS:
        if keyword.lower() in full_text.lower():
            return False

    # 3. 허용된 언어/문자 비율 검사 (외계어, 비정상 문자 차단)
    # 영어, 한국어, 숫자, 기본 특수문자만 허용
    allowed_pattern = re.compile(r"[a-zA-Z0-9가-힣\s\.,\?!'\"\(\)\[\]-]")
    allowed_char_count = len(allowed_pattern.findall(full_text))
    total_char_count = len(full_text)
    
    if total_char_count > 0:
        valid_ratio = allowed_char_count / total_char_count
        # 유효 문자가 50% 미만이면 노이즈 (예: ༱༱༱, 이모지 폭탄 등)
        if valid_ratio < 0.5: 
            return False

    # 4. 반복 패턴 검사 (같은 단어 무한 반복)
    words = full_text.split()
    if len(words) > 5:
        unique_words = set(words)
        ratio = len(unique_words) / len(words)
        # 고유 단어 비율이 20% 미만이면 삭제
        if ratio < 0.2: 
            return False

    return True

# ==========================================
# 🚀 메인 처리 (Processing)
# ==========================================
def main():
    if not os.path.exists(SRC_ROOT):
        print(f"❌ Error: Source directory '{SRC_ROOT}' not found.")
        return

    # STT 폴더 내의 모든 데이터셋 폴더 탐색 (video-mme, mvbench 등)
    dataset_dirs = [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))]
    
    print(f"📂 Found Datasets: {dataset_dirs}")
    print(f"🚀 Preprocessing STT files...")
    print(f"   - Valid   -> '{DST_ROOT}'")
    print(f"   - Filtered -> '{FILTERED_ROOT}'\n")

    total_files = 0
    kept_files = 0
    filtered_files = 0

    for dataset in dataset_dirs:
        src_dataset_path = os.path.join(SRC_ROOT, dataset)
        dst_dataset_path = os.path.join(DST_ROOT, dataset)
        filtered_dataset_path = os.path.join(FILTERED_ROOT, dataset) # 걸러진 파일 경로

        # 타겟 폴더들 생성
        os.makedirs(dst_dataset_path, exist_ok=True)
        os.makedirs(filtered_dataset_path, exist_ok=True)

        files = [f for f in os.listdir(src_dataset_path) if f.endswith(".json")]
        
        print(f"   Processing '{dataset}' ({len(files)} files)...")

        for filename in tqdm.tqdm(files, desc=f"   Filtering {dataset}", leave=False):
            total_files += 1
            src_file = os.path.join(src_dataset_path, filename)
            
            try:
                with open(src_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 🔥 필터 통과 여부 확인
                if is_valid_transcript(data):
                    # [Valid] 깨끗한 파일 저장
                    dst_file = os.path.join(dst_dataset_path, filename)
                    with open(dst_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    kept_files += 1
                else:
                    # [Filtered] 노이즈 파일 저장 (별도 폴더)
                    filtered_file = os.path.join(filtered_dataset_path, filename)
                    with open(filtered_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    filtered_files += 1

            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

    print("\n" + "="*50)
    print("✅ Preprocessing Completed!")
    print(f"📊 Total Processed  : {total_files}")
    print(f"🟢 Kept (Clean)     : {kept_files}")
    print(f"🟠 Filtered (Noise) : {filtered_files}")
    print("-" * 50)
    print(f"📁 Clean Data Path  : {DST_ROOT}")
    print(f"🗑️ Filtered Data Path : {FILTERED_ROOT}")
    print("="*50)

if __name__ == "__main__":
    main()