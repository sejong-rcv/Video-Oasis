import os
import glob
import subprocess
import shutil
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_mlvu_files(download_sub_dir="./MLVU_Test"):
    
    part_files = sorted(glob.glob(os.path.join(download_sub_dir, "test_video.tar.gz.part-*")))
    
    if not part_files:
        print("❌ [Error] No split files found to merge.")
        return

    combined_tar = os.path.join(download_sub_dir, "test_video.tar.gz")
    
    try:
        print(f"🔄 [Merging] Combining {len(part_files)} files into {combined_tar}...")
        merge_cmd = f"cat {os.path.join(download_sub_dir, 'test_video.tar.gz.part-*')} > {combined_tar}"
        subprocess.run(merge_cmd, shell=True, check=True)
        print("✅ [Success] Files merged successfully.")

        print(f"📦 [Extracting] Extracting {combined_tar}...")
        extract_cmd = f"tar -xzf {combined_tar} -C ./"
        subprocess.run(extract_cmd, shell=True, check=True)
        print("✅ [Success] Extraction complete.")

        extracted_folder = "./video"
        target_folder = "./videos"
        
        if os.path.exists(extracted_folder):
            print(f"📂 [Renaming] Renaming '{extracted_folder}' to '{target_folder}'...")
            os.rename(extracted_folder, target_folder)
            print("✅ [Success] Folder renamed to 'videos'.")
        else:
            print(f"⚠️ [Warning] Could not find the expected '{extracted_folder}'. Please check the folder names manually.")

        print("🗑️ [Cleaning up] Deleting original part files and the combined tar file...")
        for part in part_files:
            os.remove(part)
        os.remove(combined_tar)
        print("✅ [Success] Clean up complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ [Error] A command failed during execution: {e}")
    except Exception as e:
        print(f"❌ [Error] An unexpected error occurred: {e}")

if __name__ == '__main__':

    snapshot_download(
        repo_id="MLVU/MLVU_Test",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="MLVU_Test/*"
    )
    process_mlvu_files()