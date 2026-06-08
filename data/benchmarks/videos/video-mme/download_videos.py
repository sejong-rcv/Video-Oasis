import os
import zipfile
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_video_mme_zips(base_dir="./"):
    
    print(f"📂 [Info] Scanning for zip files in: {base_dir}")

    zip_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".zip")])
    
    if not zip_files:
        print("⚠️ [Warning] No zip files found to extract.")
        return

    for filename in zip_files:
        file_path = os.path.join(base_dir, filename)
        
        try:
            print(f"📦 [Extracting] {filename}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            
            print(f"🗑️ [Deleting] {filename} (Extraction complete)")
            os.remove(file_path)
            
        except zipfile.BadZipFile:
            print(f"❌ [Error] {filename} is a corrupted zip file and cannot be extracted.")
        except Exception as e:
            print(f"❌ [Error] An error occurred while processing {filename}: {e}")

    extracted_data_dir = os.path.join(base_dir, "data")
    target_videos_dir = os.path.join(base_dir, "videos")

    if os.path.exists(extracted_data_dir):
        print(f"🔄 [Renaming] Renaming '{extracted_data_dir}' to '{target_videos_dir}'...")
        try:
            os.rename(extracted_data_dir, target_videos_dir)
            print("✅ [Success] Folder successfully renamed to 'videos'.")
        except OSError as e:
            print(f"⚠️ [Warning] Could not rename folder. It might already exist: {e}")
    else:
        print("⚠️ [Warning] Expected 'data' folder was not found after extraction.")

    print("✅ [Success] All Video-MME zip files processed successfully!")


if __name__ == '__main__':

    snapshot_download(
        repo_id="lmms-lab/Video-MME",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="*.zip"
    )

    process_video_mme_zips()