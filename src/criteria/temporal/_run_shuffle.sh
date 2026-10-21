export PYTHONPATH="../..${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run frame-shuffle \
  --model qwen3_vl \
  --num-frames 128

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run frame-shuffle \
  --model eagle25 \
  --num-frames 128

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run frame-shuffle \
  --model videoauto_r1_qwen3 \
  --num-frames 128
