# Video-Oasis Diagnostic Criteria

This package keeps the paper's diagnostic structure explicit in code.

The original implementation is preserved in `src/criteria_original`. The
refactored code separates reusable infrastructure from the three diagnostic axes:

- `visual`: visual-dependency tests
  - `blind`: no auxiliary context
  - `audio`: Whisper transcript context
  - `summary`: CARE caption context
- `temporal`: temporal-dependency tests
  - `center_frame`: single middle frame
  - `frame_shuffle`: shuffled uniformly sampled frames
  - `bag_of_frames`: temporal-agnostic VLM top-k matching
- `ambiguity`: annotation-reliability checks
  - `consistency`: cross-model disagreement
  - `redundancy`: all chunks answerable
  - `sensitivity`: shuffled-success manual restoration candidates
- `aggregate_results.py`: cross-model consensus and shortcut-set construction
- `filter_annotations.py`: filtered annotation construction from shortcut ids

Shared annotation, video, prompt, output, and model-registry helpers live in
`common`.

The temporal draft is organized around two execution paths:

- `runner_mllm.py` contains the sampling and generative MLLM pipeline used by
  Center-Frame and Frame Shuffling.
- `runner_vlm.py` orchestrates Bag-of-Frames feature extraction and evaluation.
- `bag_of_frames/features.py` owns feature cache naming, extraction, lookup, and loading.
- `bag_of_frames/scoring.py` contains temporal-agnostic top-k answer scoring.

The temporal CLI uses subcommands so each test exposes only relevant options:

```bash
python -m criteria.temporal.run center-frame --model eagle25
python -m criteria.temporal.run frame-shuffle --model eagle25 --num-frames 128
python -m criteria.temporal.run bag-of-frames --model longclip
```

Center-Frame and Frame Shuffling use the shared MLLM runner. Bag-of-Frames
extracts one frame per video second (up to 2048), caches normalized frame
features, and evaluates answer options with top-k mean or max aggregation.
Feature caches are stored under model-family directories such as `clip/`,
`eva/`, and `longclip/` beneath the supplied feature root.
If the LongCLIP checkpoint is missing, it is downloaded automatically from
`BeichenZhang/LongCLIP-L` on Hugging Face before model loading.

The visual tests share one text-only execution path:

- `visual/runner_text.py` runs Blind, Audio, and Summary inference.
- `visual/contexts.py` resolves empty, Whisper transcript, and CARE summary context.
- `common/text.py` provides the text-only generation interface for MLLMs.

Audio extraction and Whisper transcription are preprocessing steps outside the
visual evaluation runner. The Audio Test reads existing transcripts from
`data/benchmarks/audios/stt`.

After all six diagnostic tests finish, aggregate their predictions with the
paper's strict three-model consensus rule:

```bash
cd /data3/gtlim/workspace/src/Video-Oasis
PYTHONPATH=src python -m criteria.aggregate_results --consensus 3
```

Each test is positive only when all three configured models produce valid
predictions and meet the consensus threshold. Audio samples without an STT
context are marked `not_applicable`; missing or unparsable predictions for an
otherwise applicable test are marked `incomplete` and never silently treated as
incorrect predictions.

Manual Frame Shuffling restorations can be supplied as one sample id per line:

```bash
PYTHONPATH=src python -m criteria.aggregate_results \
  --consensus 3 \
  --sensitivity-restore sensitivity_restore_ids.txt
```

The default report directory is `src/criteria/reports`. It contains the complete
diagnostic decisions, incomplete-result audit records, summary statistics, and
`shortcut_ids.txt`. Build the filtered annotation file with:

```bash
PYTHONPATH=src python -m criteria.filter_annotations \
  --shortcut-ids src/criteria/reports/shortcut_ids.txt \
  --output video_oasis_filtered.json
```
