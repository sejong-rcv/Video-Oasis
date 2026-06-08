import os
import zipfile
import shutil
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_mvbench_zips(base_dir="./video", target_dir="./videos"):
    
    if not os.path.exists(base_dir):
        print(f"❌ [Error] Directory not found: {base_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)
    print(f"📂 [Info] Target directory ready: {target_dir}/")
    print(f"📂 [Info] Scanning for zip files in: {base_dir}")
    
    for filename in os.listdir(base_dir):
        if filename.endswith(".zip"):
            file_path = os.path.join(base_dir, filename)
            temp_extract_dir = os.path.join(base_dir, "temp_extract")
            
            try:
                print(f"📦 [Extracting] {filename} to temporary folder...")
                os.makedirs(temp_extract_dir, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                print(f"🔄 [Moving] Extracting innermost files from {filename} to {target_dir}/...")
                for root, _, files in os.walk(temp_extract_dir):
                    for file in files:
                        extracted_file_path = os.path.join(root, file)
                        target_file_path = os.path.join(target_dir, file)
                        
                        shutil.move(extracted_file_path, target_file_path)
                
                shutil.rmtree(temp_extract_dir)
                print(f"🗑️ [Deleting] {filename} (Processing complete)")
                os.remove(file_path)
                
            except zipfile.BadZipFile:
                print(f"❌ [Error] {filename} is a corrupted zip file and cannot be extracted.")
            except Exception as e:
                print(f"❌ [Error] An error occurred while processing {filename}: {e}")

    shutil.rmtree(base_dir)
    print(f"✅ [Success] All files successfully extracted and flattened into '{target_dir}'!")


if __name__ == '__main__':
    
    snapshot_download(
        repo_id="OpenGVLab/MVBench",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="video/*"
    )

    process_mvbench_zips(base_dir="./video", target_dir="./videos")