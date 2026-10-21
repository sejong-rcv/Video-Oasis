<div align="center">

# Video-Oasis: Rethinking Evaluation of Video Understanding

<p align="center">
    <img src="assets/video-native-challenges.png" width="720" style="margin-bottom: 0.2;"/>
</p>

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
- [x] Release the [Video-Oasis](src/criteria) <br>

## Update

We corrected missing temporal boundaries in the evaluation annotations and reran the diagnostic tests on 2,163 affected questions.

| Change | Questions |
| --- | ---: |
| Previous release | 11,031 |
| Removed | -177 |
| Restored with category annotations | +409 |
| **Updated release** | **11,263** |

Updated evaluation results will be published in this repository.

# 🔥 Getting Started

## 🔨 Installation

* **Requirements:** Python ≥ 3.12, CUDA-compatible GPUs, and `torch==2.8.0`. See the table below for model-specific dependencies.

### Model Compatibility

| Model | Tested `transformers` | Additional Dependencies |
| --- | --- | --- |
| [Eagle2.5-8B](https://huggingface.co/nvidia/Eagle2.5-8B) | 4.57.0 | |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 4.57.0 | |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) / [Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | 4.57.0 | |
| [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 4.57.0 | `librosa==0.11.0`, `qwen-omni-utils==0.0.9` |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 5.5.0 | `huggingface-hub==1.5.0` |
| [VideoAuto-R1-Qwen2.5-VL-7B](https://huggingface.co/IVUL-KAUST/VideoAuto-R1-Qwen2.5-VL-7B) / [Qwen3-VL-8B](https://huggingface.co/IVUL-KAUST/VideoAuto-R1-Qwen3-VL-8B) | 4.57.0 | |
| [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B) / [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 4.57.0 | |
| [InternVideo3-8B-Instruct](https://huggingface.co/yanziang/InternVideo3-8B-Instruct) | 4.57.3 | |
| [LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) | 4.43.4 | Official LLaVA-NeXT source |
| [VidLaDA-8B](https://huggingface.co/ziHoHe/VidLaDA-8B) | 4.43.4 | VidLaDA's `llava` source |
| [MiMo-VL-7B-SFT-2508](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-SFT-2508) | 4.57.0 | |
| [VideoLLaMA3-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B) | 4.46.3 | `ffmpeg-python==0.2.0`, `future==1.0.0` |
| [Gemma4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 5.13.1 | `huggingface-hub==1.5.0` |

## 🎞 Dataset & Models

### Dataset
* We begin by curating 14 diverse benchmarks, covering tasks from perception to reasoning across durations spanning seconds to hours.

* The full list of benchmarks is available [here](https://github.com/sejong-rcv/Video-Oasis/tree/main/data/benchmarks/videos).
     * Run `python download_videos.py` within each directory to download the data.
  
* After downloading the benchmarks, run `python run_ffmpeg.py` from `data/benchmarks/videos` to check decoding and re-encode problematic videos. This cannot recover missing videos or restore footage removed from shortened source videos.
  
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

* To download checkpoints, keep only the desired `snapshot_download` calls enabled in [`data/models/download_models.py`](data/models/download_models.py), then run `python download_models.py` from `data/models`. Without editing, the script attempts to download every listed model.

* By default, we support models that can be downloaded via `huggingface_hub`'s `snapshot_download`.

* For custom models, please download them manually and place them in the appropriate directory.

##  <img src="assets/icon.png" width="30" height="30" align="center"> Video-Oasis

The diagnostic suite in [`src/criteria`](src/criteria) uses [`video_total.json`](src/lmms_eval/video_total.json) by default.

| Group | Test | Script |
| --- | --- | --- |
| Visual | Blind | [visual/_run_blind.sh](src/criteria/visual/_run_blind.sh) |
| Visual | Audio transcript | [visual/_run_audio.sh](src/criteria/visual/_run_audio.sh) |
| Visual | Summary | [visual/_run_summary.sh](src/criteria/visual/_run_summary.sh) |
| Temporal | Center frame | [temporal/_run_center.sh](src/criteria/temporal/_run_center.sh) |
| Temporal | Frame shuffling | [temporal/_run_shuffle.sh](src/criteria/temporal/_run_shuffle.sh) |
| Temporal | Bag-of-Frames | [temporal/_run_bof.sh](src/criteria/temporal/_run_bof.sh) |

Audio requires transcripts in `data/benchmarks/audios/stt`; Summary requires `src/criteria/visual/total_summary.json`.

Run scripts from their containing directories, then aggregate after all six tests finish. Examples from the repository root:

```bash
(cd src/criteria/visual && bash _run_blind.sh)
(cd src/criteria/temporal && bash _run_center.sh)
# After completing all six tests:
(cd src && python -m criteria.aggregate_results --consensus 3)
```

Within each test, all three models must answer correctly to flag a shortcut. Questions flagged by any test are collected in `src/criteria/reports/shortcut_ids.txt`.

[Ambiguity checks](src/criteria/ambiguity) (Consistency, Redundancy, and manual Sensitivity review) and category annotation are separate steps before finalizing the dataset.

## 📑 Evaluation

Evaluation uses the bundled [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), with model-specific scripts in [`src/scripts`](src/scripts).

Scripts default to `v_oasis` (the final Video-Oasis questions). Set `task=vqa_total` to evaluate the full merged benchmark set instead.

Before running, update `dataset_path`, `dataset_kwargs.data_files`, and any configured `cache_dir` in the selected task YAML ([v_oasis](src/lmms_eval/tasks/v_oasis/v_oasis.yaml) or [vqa_total](src/lmms_eval/tasks/vqa_total/vqa_total.yaml)) to match your local paths.

Run from the repository root:

```bash
cd src
bash scripts/eval_eagle25_8B.sh
```

Adjust the checkpoint path and GPU process count in the script for your environment; results are saved under `src/experiments` by default.

---

# Acknowledgements 👍

* Source code is built upon [VideoAuto-R1](https://github.com/IVUL-KAUST/VideoAuto-R1). 

* Evaluation is powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). 

* We extend our gratitude to the creators of the following pioneering benchmarks, which laid the foundation for our work: [EgoSchema](https://github.com/egoschema/EgoSchema), [ImplicitQA](https://github.com/UCF-CRCV/VRR-QA), [LongVideoBench](https://github.com/longvideobench/LongVideoBench), [LVBench](https://github.com/zai-org/LVBench), [MINERVA](https://github.com/google-deepmind/neptune?tab=readme-ov-file), [MLVU](https://github.com/JUNJIE99/MLVU), [MMR-V](https://github.com/GaryStack/MMR-V), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [RTV-Bench](https://github.com/LJungang/RTV-Bench), [TVBench](https://github.com/daniel-cores/tvbench), [VCR-Bench](https://github.com/zhishuifeiqian/VCR-Bench), [Video-MME](https://github.com/MME-Benchmarks/Video-MME), [Video-Holmes](https://github.com/TencentARC/Video-Holmes), and [VSI-bench](https://github.com/vision-x-nyu/thinking-in-space).

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
