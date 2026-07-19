<div align="center">

# Video-Oasis: Rethinking Evaluation of Video Understanding

<p align="center">
    <img src="assets/video-native-challenges.png" width="720" style="margin-bottom: 0.2;"/>
<p>

</div>

<div align="center">

[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-Visit-blue)](https://limgeuntaekk.github.io/Video-Oasis/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.29616-b31b1b.svg)](https://arxiv.org/abs/2603.29616)
[![GitHub](https://img.shields.io/badge/💻%20GitHub-Code-green)](https://github.com/sejong-rcv/Video-Oasis)
[![Hugging Face Paper](https://img.shields.io/badge/Daily%20Paper-Hugging%20Face-f59e0b?logo=huggingface)](https://huggingface.co/papers/2603.29616)
</div>

> **TL;DR.** Video-Oasis rethinks the current benchmark landscape by examining whether proliferating video benchmarks truly satisfy shared criteria for genuine video understanding.

# News
- [x] Release the paper on <a href="https://arxiv.org/abs/2603.29616">arXiv</a> <br>
- [x] Release the Video-Native Challenges on <a href="https://github.com/sejong-rcv/Video-Oasis/blob/main/src/lmms_eval/video_oasis.json">link</a> <br>
- [x] Release the [Video-Oasis diagnostic suite](src/criteria) <br>

# 🔥 Getting Started

## 🔨 Installation

* **Requirements:** Python ≥ 3.12, CUDA-compatible GPUs, `torch`, `vllm >= 0.11.0`, `transformers >= 4.57.0`.

## 🎞 Dataset & Models

### Dataset
* We begin by curating 14 diverse benchmarks, covering tasks from perception to reasoning across durations spanning seconds to hours.

* The full list of benchmarks is available [here](https://github.com/sejong-rcv/Video-Oasis/tree/main/data/benchmarks/videos).
     * Run `python download_videos.py` within each directory to download the data.
  
* After downloading all benchmarks, run `python run_ffmpeg.py` to process and fix any corrupted or unreadable videos.
  
* Once completed, your directory structure should look like this:
~~~~
├── data/benchmarks/videos
   ├── egoschema
      └── videos
         ├── video_1
         ├── video_2
         └── ...

   ├── implicitqa
      └── videos
         ├── video_1
         ├── video_2
         └── ...

   ├── ...
 
   ├── vsi-bench
      └── videos
         ├── video_1
         ├── video_2
         └── ...
~~~~
### Model

* For model checkpoint, move to the ```data/models``` directory and run ```python download_models.py``` to download your desired models.

* By default, we support models that can be downloaded via `huggingface_hub`'s `snapshot_download`.

* For custom models, please download them manually and place them in the appropriate directory.

## 📑 Evaluation

* Evaluation is handled via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), which is bundled within this repository.
  
* The scripts to run the benchmark suite used in our paper are located in the `src/scripts` directory.
  
* To execute the evaluation, simply set the task argument to either `vqa_total` or `v_oasis` in the script.

* An example execution script is provided below:

```bash
model_path=Video-Oasis/data/models/models/Eagle2.5-8B
output_path=./experiments/Eagle2.5-8B_16K/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

task=vqa_total # or v_oasis

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model eagle2_5 \
    --model_args pretrained=$model_path, \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/"
```

##  <img src="assets/icon.png" width="30" height="30" align="center"> Video-Oasis

The Video-Oasis diagnostic suite is available in [`src/criteria`](src/criteria).
It includes the Visual Dependency, Temporal Dependency, and Ambiguity tests used
to construct Video-Oasis. See the [diagnostic suite guide](src/criteria/README.md)
for the code structure and execution workflow.

---

# Acknowledgements 👍

* Source code is built upon [VideoAuto-R1](https://github.com/IVUL-KAUST/VideoAuto-R1). 

* Evaluation is powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). 

* We extend our gratitude to the creators of the following pioneering benchmarks, which laid the foundation for our work: [EgoShema](https://github.com/egoschema/EgoSchema), [ImplicitQA](https://github.com/UCF-CRCV/VRR-QA), [LongVideoBench](https://github.com/longvideobench/LongVideoBench), [LVBench](https://github.com/zai-org/LVBench), [MINERVA](https://github.com/google-deepmind/neptune?tab=readme-ov-file), [MLVU](https://github.com/JUNJIE99/MLVU), [MMR-V](https://github.com/GaryStack/MMR-V), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [RTV-Bench](https://github.com/LJungang/RTV-Bench), [TVBench](https://github.com/daniel-cores/tvbench), [VCR-Bench](https://github.com/zhishuifeiqian/VCR-Bench), [Video-MME](https://github.com/MME-Benchmarks/Video-MME), [Video-Holmes](https://github.com/TencentARC/Video-Holmes), and [VSI-bench](https://github.com/vision-x-nyu/thinking-in-space). 

# Citation 🎓

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{lim2026videooasis,
  title={Video-Oasis: Rethinking Evaluation of Video Understanding},
  author={Geuntaek Lim and Sungjune Park and Jaeyun Lee and Inwoong Lee and Taeoh Kim and Dongyoon Wee and Minho Shim and Yukyung Choi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

# License 📄

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
