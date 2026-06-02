---
language:
- en
license: mit
size_categories:
- 1K<n<10K
task_categories:
- video-text-to-text
pretty_name: ImplicitQA
configs:
- config_name: default
  data_files:
  - split: eval
    path: ImplicitQAv0.1.2.jsonl
tags:
- implicit-reasoning
---

<h1 align="center"> ImplicitQA: Going beyond frames towards Implicit Video Reasoning </h1>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="">
</p>

<p align="center">
  <a href="https://swetha5.github.io/">Sirnam Swetha </a> &nbsp;|&nbsp;
  <a href="https://www.rohitg.xyz/">Rohit Gupta </a> &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/parth-parag-kulkarni-739302150">Parth Parag Kulkarni </a> &nbsp;|&nbsp;
  <a href="https://davidshatwell.com/">David G Shatwell </a> &nbsp;|&nbsp;
  <a href="https://jachansantiago.com/">Jeffrey A Chan Santiago </a> &nbsp;|&nbsp;
  <a href="https://nylesiddiqui.github.io/">Nyle Siddiqui </a> &nbsp;|&nbsp;
  <a href="https://joefioresi718.github.io/">Joseph Fioresi </a> &nbsp;|&nbsp;
  <a href="https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao">Mubarak Shah</a> <br><br>
  University of Central Florida&emsp;
</p>

<div align="center">

[![](https://img.shields.io/badge/Project%20Page-ab99d7)](https://swetha5.github.io/ImplicitQA/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.21742-b31b1b.svg)](https://arxiv.org/abs/2506.21742)&nbsp;
[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ImplicitQA-orange)](https://huggingface.co/datasets/ucf-crcv/ImplicitQA)&nbsp;

---
</div>

# ImplicitQA Dataset

The ImplicitQA dataset was introduced in the paper [ImplicitQA: Going beyond frames towards Implicit Video Reasoning](https://arxiv.org/abs/2506.21742).

**Project page:** https://swetha5.github.io/ImplicitQA/

ImplicitQA is a novel benchmark specifically designed to test models on **implicit reasoning** in Video Question Answering (VideoQA). Unlike existing VideoQA benchmarks that primarily focus on questions answerable through explicit visual content (actions, objects, events directly observable within individual frames or short clips), ImplicitQA addresses the need for models to infer motives, causality, and relationships across discontinuous frames. This mirrors human-like understanding of creative and cinematic videos, which often employ storytelling techniques that deliberately omit certain depictions.

The dataset comprises 1,000 meticulously annotated QA pairs derived from 1,000 high-quality creative video clips. These QA pairs are systematically categorized into key reasoning dimensions, including:

*   Lateral vertical spatial reasoning
*   Vertical spatial reasoning
*   Relative Depth and proximity
*   Viewpoint and visibility
*   Motion and trajectory Dynamics
*   Causal and motivational reasoning
*   Social interactions and Relations
*   Physical and Environmental context
*   Inferred counting

The annotations are deliberately challenging, crafted to ensure high quality and to highlight the difficulty of implicit reasoning for current VideoQA models. By releasing both the dataset and its data collection framework, the authors aim to stimulate further research and development in this crucial area of AI.

---

## Citation
If you use this dataset and/or this code in your work, please cite our [paper](https://arxiv.org/abs/2506.21742):

```bibtex
@article{swetha2025implicitqa,
title={ImplicitQA: Going beyond frames towards Implicit Video Reasoning},
author={Swetha, Sirnam and Gupta, Rohit and Kulkarni, Parth Parag and Shatwell, David G and Santiago, Jeffrey A Chan and Siddiqui, Nyle and Fioresi, Joseph and Shah, Mubarak},
journal={arXiv preprint arXiv:2506.21742},
year={2025}
}
```