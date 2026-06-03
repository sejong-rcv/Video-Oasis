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


# News
- [x] Release the paper on <a href="https://arxiv.org/abs/2603.29616">arXiv</a> <br>
- [x] Release the Video-Native Challenges on <a href="https://github.com/sejong-rcv/Video-Oasis/blob/main/src/lmms_eval/video_oasis.json">link</a> <br>
- [ ] Release the code for Video-Oasis <br>

# Key Findings

<p align="center">
    <img src="assets/motivation.png" width="1080" style="margin-bottom: 0.2;"/>
<p>

**RL models dive depth, base models seek breadth.** When limited to a single attempt, RL models trained with GRPO generally outperform their base counterparts. However, when multiple samplings are permitted, base models consistently succeed in solving a broader range of problems by leveraging diverse and alternative reasoning pathways that RL models have discarded.

**Diversity collapse is the culprit.** During early GRPO training (within the first 20 steps), reasoning diversity drops sharply to a negligible level. The model prematurely converges on a narrow set of strategies while abandoning the vast majority of potential alternatives — before it has even seen most of the training data.

**Divergent thinking increases the odds of success.** Across all evaluated benchmarks, there is a strong positive correlation between reasoning diversity and acc@4. Tackling a problem through diverse strategies, rather than adhering to a single mode, significantly facilitates the discovery of correct answers.


# Getting Started 🔥

## Installation 🔨

**Requirements:** Python ≥ 3.12, CUDA-compatible GPUs, `torch`, `vllm >= 0.11.0`, `transformers >= 4.57.0`.

```bash
git clone https://github.com/sejong-rcv/Video-Oasis.git
cd Video-Oasis
pip install -e .
```

## Dataset 🎞 

We release the MUPO training data on Hugging Face, curated from [ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K).


## Evaluation 📑

Evaluation is handled via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), which is bundled in this repo. To run the benchmark suite used in the paper:

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

# Acknowledgements 👍

* Source code is built upon [VideoAuto-R1](https://github.com/IVUL-KAUST/VideoAuto-R1). 

* Evaluation is powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). 

* We extend our gratitude to the creators of the following pioneering benchmarks, which laid the foundation for our work: [EgoShema](https://github.com/egoschema/EgoSchema), [ImplicitQA](https://github.com/UCF-CRCV/VRR-QA), [LongVideoBench](https://github.com/longvideobench/LongVideoBench), [LVBench](https://github.com/zai-org/LVBench), [MINERVA](https://github.com/google-deepmind/neptune?tab=readme-ov-file), [MLVU](https://github.com/JUNJIE99/MLVU), [MMR-V](https://github.com/GaryStack/MMR-V), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [RTV-Bench](https://github.com/LJungang/RTV-Bench), [TVBench](https://github.com/daniel-cores/tvbench), [VCR-Bench](https://github.com/zhishuifeiqian/VCR-Bench), [Video-MME](https://github.com/MME-Benchmarks/Video-MME), [Video-Holmes](https://github.com/TencentARC/Video-Holmes), and [VSI-bench](https://github.com/vision-x-nyu/thinking-in-space). 

# Citation 🎓

If you find this work useful, please cite our paper:

```bibtex
@article{lim2026video,
  title={Video-Oasis: Rethinking Evaluation of Video Understanding},
  author={Lim, Geuntaek and Shim, Minho and Park, Sungjune and Lee, Jaeyun and Lee, Inwoong and Kim, Taeoh and Wee, Dongyoon and Choi, Yukyung},
  journal={arXiv preprint arXiv:2603.29616},
  year={2026}
}
```

# License 📄

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
