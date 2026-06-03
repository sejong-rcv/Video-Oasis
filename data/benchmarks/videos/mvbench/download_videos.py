import os
import zipfile
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_mvbench_zips(base_dir="./video"):
    """video 폴더 내의 각 zip 파일을 자신의 이름과 같은 폴더에 압축 해제하고 삭제합니다."""
    
    if not os.path.exists(base_dir):
        print(f"❌ [Error] Directory not found: {base_dir}")
        return

    print(f"📂 [Info] Scanning for zip files in: {base_dir}")
    
    for filename in os.listdir(base_dir):
        if filename.endswith(".zip"):
            file_path = os.path.join(base_dir, filename)
            
            # zip 파일 이름에서 확장자(.zip)를 제거하여 새 폴더 이름 생성
            # (예: clevrer.zip -> clevrer)
            folder_name = os.path.splitext(filename)[0]
            extract_dir = os.path.join(base_dir, folder_name)
            
            try:
                # 압축을 풀 전용 폴더 생성
                os.makedirs(extract_dir, exist_ok=True)
                
                # 압축 해제
                print(f"📦 [Extracting] {filename} -> {extract_dir}/ ...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
                # 압축 해제가 성공적으로 끝나면 원본 zip 파일 삭제
                print(f"🗑️ [Deleting] {filename} (Extraction complete)")
                os.remove(file_path)
                
            except zipfile.BadZipFile:
                print(f"❌ [Error] {filename} is a corrupted zip file and cannot be extracted.")
            except Exception as e:
                print(f"❌ [Error] An error occurred while processing {filename}: {e}")

    print("✅ [Success] All zip files processed successfully!")

if __name__ == '__main__':

    snapshot_download(
        repo_id="OpenGVLab/MVBench",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="video/*"
    )

    process_mvbench_zips(base_dir="./video")