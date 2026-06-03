import os
import zipfile
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def extract_and_delete_zips(directory='./'):
    target_dir = os.path.join(directory)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📂 [Info] Extracting zip files to: {target_dir}")

    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            file_path = os.path.join(directory, filename)
            
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

    print("✅ [Success] All zip files processed successfully!")

if __name__ == '__main__':

    snapshot_download(
        repo_id="lmms-lab/egoschema",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="*.zip"
    )
    print()
    
    extract_and_delete_zips()