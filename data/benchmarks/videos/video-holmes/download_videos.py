import os
import zipfile
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_video_holmes_zip(base_dir="./"):
    
    zip_file_path = os.path.join(base_dir, "videos.zip")
    extracted_folder = os.path.join(base_dir, "videos_cropped")
    target_folder = os.path.join(base_dir, "videos")

    if not os.path.exists(zip_file_path):
        print("⚠️ [Warning] videos.zip not found in the current directory.")
        return

    try:
        print(f"📦 [Extracting] videos.zip...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        if os.path.exists(extracted_folder):
            print(f"🔄 [Renaming] Renaming '{extracted_folder}' to '{target_folder}'...")
            try:
                os.rename(extracted_folder, target_folder)
                print("✅ [Success] Folder successfully renamed to 'videos'.")
            except OSError as e:
                print(f"⚠️ [Warning] Could not rename folder. The 'videos' folder might already exist: {e}")
        else:
            print(f"⚠️ [Warning] Expected '{extracted_folder}' was not found after extraction. Please check the zip contents.")

        print(f"🗑️ [Deleting] videos.zip (Extraction complete)")
        os.remove(zip_file_path)
        print("✅ [Success] Clean up complete!")

    except zipfile.BadZipFile:
        print("❌ [Error] videos.zip is corrupted and cannot be extracted.")
    except Exception as e:
        print(f"❌ [Error] An unexpected error occurred: {e}")

if __name__ == '__main__':

    snapshot_download(
        repo_id="TencentARC/Video-Holmes",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="videos.zip"
    )

    process_video_holmes_zip()