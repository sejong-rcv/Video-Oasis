# Video-Oasis Diagnostic Criteria

This directory implements the diagnostic suite used to construct Video-Oasis.

| Axis | Test | Models |
| --- | --- | --- |
| Visual | Blind, Audio, Summary | Qwen2.5-VL, Qwen3-VL, Eagle2.5 |
| Temporal | Center Frame, Frame Shuffling | Qwen2.5-VL, Qwen3-VL, Eagle2.5 |
| Temporal | Bag of Frames | CLIP, EVA-CLIP, LongCLIP |
| Ambiguity | Consistency | Five existing full-video prediction files |
| Ambiguity | Redundancy | Eagle2.5 on 8 chunks of 16 frames |
| Ambiguity | Sensitivity | Manual review of Frame Shuffling positives |

Visual and Temporal tests use `src/lmms_eval/video_total.json` by default.

## Run Diagnostics

The shell scripts use paths relative to their own directories. Run them as
follows:

```bash
cd src/criteria/visual
bash _run_blind.sh
bash _run_audio.sh
bash _run_summary.sh

cd ../temporal
bash _run_center.sh
bash _run_shuffle.sh
bash _run_bof.sh
```

Audio reads existing Whisper transcripts from `data/benchmarks/audios/stt` and
skips missing transcripts; STT preprocessing is not included. Summary uses
`src/criteria/visual/total_summary.json`. Each script uses GPU 0 by default.

## Aggregate Shortcuts

From the repository root, aggregate the six Visual and Temporal tests using the
strict three-model consensus:

```bash
PYTHONPATH=src python -m criteria.aggregate_results --consensus 3
```

Reports are written to `src/criteria/reports`.

Build the Sensitivity review queue from positive Frame Shuffling results:

```bash
PYTHONPATH=src python -m criteria.ambiguity.run sensitivity
```

Review `src/criteria/ambiguity/output/sensitivity/sensitivity_candidates.json`
and prepare `sensitivity_restore_ids.txt` with one restored sample ID per line.
Then rerun aggregation:

```bash
PYTHONPATH=src python -m criteria.aggregate_results \
  --consensus 3 \
  --sensitivity-restore sensitivity_restore_ids.txt
```

Create the shortcut-filtered annotation used by the remaining Ambiguity tests:

```bash
PYTHONPATH=src python -m criteria.filter_annotations \
  --shortcut-ids src/criteria/reports/shortcut_ids.txt \
  --output src/lmms_eval/video_oasis.json
```

## Run Ambiguity Tests

Consistency and Redundancy default to `src/lmms_eval/video_oasis.json`:

```bash
cd src/criteria/ambiguity
bash _run_consistency.sh
bash _run_redundancy.sh
```

Consistency uses maximal disagreement: the number of unique predictions must
equal `min(number of options, 5)`. Redundancy marks a sample when all 8 chunk
predictions are valid and correct. Both produce candidates for manual review;
they do not automatically remove annotations.

After review, place confirmed IDs in `ambiguity_exclude_ids.txt` and build the
final annotation:

```bash
cd ../../..
PYTHONPATH=src python -m criteria.filter_annotations \
  --shortcut-ids src/criteria/reports/shortcut_ids.txt \
  --exclude-ids ambiguity_exclude_ids.txt \
  --output src/lmms_eval/video_oasis_filtered.json
```
