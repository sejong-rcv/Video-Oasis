import os
import glob
import subprocess
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

def process_split_tar_files(directory="./"):
    
    part_files = sorted(glob.glob(os.path.join(directory, "videos.tar.part.*")))
    
    if not part_files:
        print("❌ [Error] No split files found to merge.")
        return

    combined_tar = os.path.join(directory, "videos.tar")
    
    videos_dir = os.path.join(directory, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    try:
        print(f"🔄 [Merging] Combining {len(part_files)} files into {combined_tar}...")
        merge_cmd = f"cat {os.path.join(directory, 'videos.tar.part.*')} > {combined_tar}"
        subprocess.run(merge_cmd, shell=True, check=True)
        print("✅ [Success] Files merged successfully.")

        print(f"📦 [Extracting] Extracting {combined_tar} to {videos_dir}...")
        extract_cmd = f"tar -xf {combined_tar} -C {videos_dir}"
        subprocess.run(extract_cmd, shell=True, check=True)
        print("✅ [Success] Extraction complete.")

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
        repo_id="JokerJan/MMR-VBench",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="videos.tar.part.*"
    )

    process_split_tar_files()