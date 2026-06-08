import os
import zipfile
import shutil
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_tvbench_zips(base_dir="./video", target_dir="./videos"):
    if not os.path.exists(base_dir):
        print(f"❌ [Error] Directory not found: {base_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)
    print(f"📂 [Info] Target directory ready: {target_dir}/")
    
    zip_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))

    if not zip_files:
        print("⚠️ [Warning] No zip files found.")
        return

    for file_path in zip_files:
        filename = os.path.basename(file_path)
        zip_name = os.path.splitext(filename)[0]  # 예: "1.1"

        try:
            if "egocentric_sequence" in file_path:
                print(f"📦 [Extracting & Preserving] {filename} ...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    top_level = {item.split('/')[0] for item in zip_ref.namelist()}
                    if zip_name in top_level:
                        zip_ref.extractall(target_dir)
                    else:
                        specific_target = os.path.join(target_dir, zip_name)
                        os.makedirs(specific_target, exist_ok=True)
                        zip_ref.extractall(specific_target)

            else:
                print(f"📦 [Extracting & Flattening] {filename} ...")
                temp_extract_dir = os.path.join(base_dir, "temp_extract")
                os.makedirs(temp_extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                for r, _, files in os.walk(temp_extract_dir):
                    for file in files:
                        extracted_file_path = os.path.join(r, file)
                        target_file_path = os.path.join(target_dir, file)
                        
                        if os.path.exists(target_file_path):
                            os.remove(target_file_path)
                        shutil.move(extracted_file_path, target_file_path)
                
                shutil.rmtree(temp_extract_dir)

            print(f"🗑️ [Deleting] {filename} (Complete)")
            os.remove(file_path)

        except zipfile.BadZipFile:
            print(f"❌ [Error] {filename} is corrupted.")
        except Exception as e:
            print(f"❌ [Error] {filename}: {e}")

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    print(f"✅ [Success] All files successfully processed into '{target_dir}'!")

if __name__ == '__main__':

    snapshot_download(
        repo_id="FunAILab/TVBench",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="video/*"
    )

    process_tvbench_zips(base_dir="./video", target_dir="./videos")