# 🎞 Data Preparation
## Getting the Dataset
* Download the Video QA Benchmark you want. (Hugging face supported)
```python

import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5"

# {'ImplicitQA', 'VCR-Bench', 'RTV-Bench', 'CG-Bench', 'LongVideoBench', 'VSI-Bench', 'Video-MME', 'MINERVA', 'LVBench', 'MLVU_Test'}

if __name__ == '__main__':
    snapshot_download("lmms-lab/Video-MME", repo_type='dataset', local_dir="./video-mme")
    ...
```

* [LVBench](https://github.com/zai-org/LVBench), [ImplicitQA](https://github.com/UCF-CRCV/VRR-QA), and [MINERVA](https://github.com/google-deepmind/neptune) require direct downloads from YouTube.

* We have provided a download script to facilitate this.
     * [LVBench](https://github.com/zai-org/LVBench)
     * [ImplicitQA](https://github.com/UCF-CRCV/VRR-QA)
     * [MINERVA](https://github.com/google-deepmind/neptune) 


## Getting the Models
* Download the Video-LLMs you want. (Hugging face supported)
```python

import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "5"


if __name__ == '__main__':
    snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct", local_dir="./Qwen2.5-VL-7B-Instruct", local_dir_use_symlinks=False, resume_download=True)
    ...
```

## Preprocess


## Overal Structure




~~~~
├── benchmarks
   ├── annos
      └── dataset 1
      └── dataset 2
      └── dataset 3
      └── ...
   ├── audio
      └── dataset 1
      └── dataset 2
      └── dataset 3
      └── ...
   ├── videos
      └── dataset 1
      └── dataset 2
      └── dataset 3
      └── ...
~~~~
  

