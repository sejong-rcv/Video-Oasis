<div align="center">

# Video-Oasis: Rethinking Evaluation for Video Understanding

<p align="center">
    <img src="assets/video-native-challenges.png" width="720" style="margin-bottom: 0.2;"/>
<p>

</div>

<div align="center">

[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-Visit-blue)](https://limgeuntaekk.github.io/Video-Oasis/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.29616-b31b1b.svg)](https://arxiv.org/abs/2603.29616)
[![GitHub](https://img.shields.io/badge/💻%20GitHub-Code-green)](https://github.com/sejong-rcv/Video-Oasis)

</div>


> **TL;DR.** Video-Oasis rethinks the current benchmark landscape by examining whether proliferating video benchmarks truly satisfy shared criteria for genuine video understanding.

This is the official implementation of the paper 'Video-Oasis: Rethinking Evaluation for Video Understanding'.



## Release
- [x] Release the paper on <a href="https://arxiv.org/abs/2603.29616">arXiv</a> <br>
- [x] Release the Video-Native Challenges on <a href="https://github.com/sejong-rcv/Video-Oasis/blob/main/src/lmms_eval/video_oasis.json">link</a> <br>
- [ ] Release the code for Video-Oasis <br>

# Key Findings🔍




# Getting Started🚀

## Installation

**Requirements:** Python ≥ 3.9, CUDA-compatible GPUs, `torch`, `vllm >= 0.8.0`, `transformers >= 4.51.0`.

```bash
git clone https://github.com/xytian1008/MUPO.git
cd MUPO
pip install -e .
```

## Dataset

We release the MUPO training data on Hugging Face, curated from [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K).


## Evaluation

Evaluation is handled via [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), which is bundled in this repo. To run the benchmark suite used in the paper:

```bash
python run.py \
    --model MUPO-Thinker-7B \
    --data MMStar HallusionBench MMVet MathVerse MathVista MathVision LogicVista WeMath Geometry3K
```

To reproduce the acc@4 results (parallel sampling evaluation):

```bash
python run.py \
    --model MUPO-Thinker-7B \
    --data MathVerse MathVista \
    --nshot 4 \
    --temperature 1.0
```

---

# Acknowledgements🥰

Our training framework is built upon [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl). Evaluation is powered by [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We train on [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) and evaluate on MathVerse, MathVista, MathVision, LogicVista, WeMath, Geometry3K, MMStar, HallusionBench, and MMVet. We gratefully acknowledge the computational support provided during this research.

# Citation🎓

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{tian2026allroads,
  title     = {All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models},
  author    = {Tian, Xinyu and Zou, Shu and Yang, Zhaoyuan and He, Mengqi and Tu, Peter and Zhang, Jing},
  booktitle = {CVPR},
  year      = {2026}
}
```

# License📄

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
