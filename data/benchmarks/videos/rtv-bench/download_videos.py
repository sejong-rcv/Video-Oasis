import os
import shutil
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def flatten_videos_directory(base_dir="./videos"):
    
    if not os.path.exists(base_dir):
        print(f"⚠️ [Warning] Directory not found: {base_dir}")
        return

    print(f"📂 [Info] Flattening subdirectories in: {base_dir}/")

    for item in os.listdir(base_dir):
        sub_dir_path = os.path.join(base_dir, item)
        
        if os.path.isdir(sub_dir_path):
            print(f"🔄 [Processing] Moving files from '{item}/' to 'videos/' ...")
            
            for filename in os.listdir(sub_dir_path):
                file_path = os.path.join(sub_dir_path, filename)
                
                if os.path.isfile(file_path):
                    target_path = os.path.join(base_dir, filename)
                    shutil.move(file_path, target_path)
            
            try:
                os.rmdir(sub_dir_path)
                print(f"🗑️ [Deleted] Empty folder: {item}")
            except OSError as e:
                print(f"⚠️ [Warning] Could not delete '{item}'. It might not be empty: {e}")

    print("✅ [Success] All videos have been successfully flattened into the 'videos' folder!")

if __name__ == '__main__':

    snapshot_download(
        repo_id="RTVBench/RTV-Bench",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="videos/*"
    )

    flatten_videos_directory()