from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5" 

if __name__ == '__main__':

    snapshot_download("VLM-Reasoning/VCR-Bench", repo_type='dataset', local_dir="./videos/vcrbench")
    snapshot_download("RTVBench/RTV-Bench", repo_type='dataset', local_dir="./videos/rtvbench")
    snapshot_download("CG-Bench/CG-Bench", repo_type='dataset', local_dir="./videos/cgbench")
    snapshot_download("longvideobench/LongVideoBench", repo_type='dataset', local_dir="./videos/longvideobench")
    snapshot_download("nyu-visionx/VSI-Bench", repo_type='dataset', local_dir="./videos/vsibench")
    snapshot_download("lmms-lab/Video-MME", repo_type='dataset', local_dir="./videos/videomme")
    snapshot_download("zai-org/LVBench", repo_type='dataset', local_dir="./videos/lvbench")
    snapshot_download("MLVU/MLVU_Test", repo_type='dataset', local_dir="./videos/mlvu")
    snapshot_download("JokerJan/MMR-VBench", repo_type='dataset', local_dir="./videos/mmrvbench")
    snapshot_download("OpenGVLab/MVBench", repo_type='dataset', local_dir="./videos/mvbench")
    snapshot_download("FunAILab/TVBench", repo_type='dataset', local_dir="./videos/tvbench")
    snapshot_download("TencentARC/Video-Holmes", repo_type='dataset', local_dir="./videos/videoholmes")
    snapshot_download("lmms-lab/egoschema", repo_type='dataset', local_dir="./videos/egoschema")



