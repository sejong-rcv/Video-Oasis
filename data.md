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


## QA Format

* All provided `.json` files already follow this format. However, when adding a new benchmark, you must ensure it adheres to the structure below:

```json
{
    "question": "<String: The question to be asked>",
    "video": "<String: The video file name (e.g., video_id.mp4)>",
    "options": [
        "A. <String: First option text>",
        "B. <String: Second option text>",
        "C. <String: Third option text>",
        "D. <String: Fourth option text>"
    ],
    "answer": "<String: Correct option letter (e.g., 'A', 'B', 'C', 'D')>",
    "answer_text": "<String: The exact text of the correct answer>",
    "db": "<String: Name of the source benchmark (e.g., ImplicitQA, LVBench, MINERVA)>",
    "qid": "<String: Unique identifier for the question>",
    "video_path": "<String: Absolute path to the video file>",
}
```

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
  

