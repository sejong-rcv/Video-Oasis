import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

if __name__ == '__main__':

    snapshot_download(
        repo_id="VLM-Reasoning/VCR-Bench",
        repo_type="dataset",
        local_dir="./",
        allow_patterns="v1/*"
    )