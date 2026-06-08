import os
import zipfile
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_vsi_bench_zips(base_dir="./"):
    
    target_dir = os.path.join(base_dir, "videos")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📂 [Info] Extracting zip files to: {target_dir}/")

    zip_files = [f for f in os.listdir(base_dir) if f.endswith(".zip")]
    
    if not zip_files:
        print("⚠️ [Warning] No zip files found to extract.")
        return

    for filename in zip_files:
        file_path = os.path.join(base_dir, filename)
        
        try:
            print(f"📦 [Extracting] {filename}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            print(f"🗑️ [Deleting] {filename} (Extraction complete)")
            os.remove(file_path)
            
        except zipfile.BadZipFile:
            print(f"❌ [Error] {filename} is a corrupted zip file and cannot be extracted.")
        except Exception as e:
            print(f"❌ [Error] An error occurred while processing {filename}: {e}")

    print("✅ [Success] All VSI-Bench zip files processed successfully!")

if __name__ == '__main__':

    snapshot_download(
        repo_id="nyu-visionx/VSI-Bench",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="*.zip"
    )
    process_vsi_bench_zips()