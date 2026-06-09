from huggingface_hub import snapshot_download

if __name__ == '__main__':
    snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct", local_dir="./Qwen2.5-VL-7B-Instruct",local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="Qwen/Qwen3-VL-8B-Instruct", local_dir="./Qwen3-VL-8B-Instruct",local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="Qwen/Qwen3-VL-8B-Thinking", local_dir="./Qwen3-VL-8B-Thinking",local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="nvidia/Eagle2.5-8B", local_dir="./Eagle2.5-8B",local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="Video-R1/Video-R1-7B", local_dir="./Video-R1-7B",local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="Efficient-Large-Model/LongVILA-R1-7B", local_dir="./LongVILA-R1-7B",local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B", local_dir="./VideoAuto-R1-Qwen2.5-VL-7B", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="IVUL-KAUST/VideoAuto-R1-Qwen3-VL-8B", local_dir="./VideoAuto-R1-Qwen3-VL-8B", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="OpenGVLab/InternVL3-8B", local_dir="./InternVL3-8B", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="OpenGVLab/InternVL3_5-8B", local_dir="./InternVL3_5-8B", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="meta-llama/Llama-3.1-8B", local_dir="./Llama-3.1-8B", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="Qwen/Qwen3-8B", local_dir="./Qwen3-8B", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", local_dir="./Mistral-7B-Instruct-v0.3", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="MCG-NJU/CaRe-7B", local_dir="./CaRe-7B", local_dir_use_symlinks=False, resume_download=True)
    snapshot_download(repo_id="Qwen/Qwen3.5-9B", local_dir="./Qwen3.5-9B",local_dir_use_symlinks=False, resume_download=True)
