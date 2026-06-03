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


## Release
- [x] Release the paper on <a href="https://arxiv.org/abs/2603.29616">arXiv</a> <br>
- [x] Release the Video-Native Challenges on <a href="https://github.com/sejong-rcv/Video-Oasis/blob/main/src/lmms_eval/video_oasis.json">link</a> <br>
- [ ] Release the code for Video-Oasis <br>

# Key Findings🔍

<p align="center">
    <img src="assets/motivation.png" width="1080" style="margin-bottom: 0.2;"/>
<p>

**RL models dive depth, base models seek breadth.** When limited to a single attempt, RL models trained with GRPO generally outperform their base counterparts. However, when multiple samplings are permitted, base models consistently succeed in solving a broader range of problems by leveraging diverse and alternative reasoning pathways that RL models have discarded.

**Diversity collapse is the culprit.** During early GRPO training (within the first 20 steps), reasoning diversity drops sharply to a negligible level. The model prematurely converges on a narrow set of strategies while abandoning the vast majority of potential alternatives — before it has even seen most of the training data.

**Divergent thinking increases the odds of success.** Across all evaluated benchmarks, there is a strong positive correlation between reasoning diversity and acc@4. Tackling a problem through diverse strategies, rather than adhering to a single mode, significantly facilitates the discovery of correct answers.


# Getting Started🚀

## Installation

**Requirements:** Python ≥ 3.12, CUDA-compatible GPUs, `torch`, `vllm >= 0.11.0`, `transformers >= 4.57.0`.

```bash
git clone https://github.com/sejong-rcv/Video-Oasis.git
cd Video-Oasis
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
@article{lim2026video,
  title={Video-Oasis: Rethinking Evaluation of Video Understanding},
  author={Lim, Geuntaek and Shim, Minho and Park, Sungjune and Lee, Jaeyun and Lee, Inwoong and Kim, Taeoh and Wee, Dongyoon and Choi, Yukyung},
  journal={arXiv preprint arXiv:2603.29616},
  year={2026}
}
```

# License📄

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
