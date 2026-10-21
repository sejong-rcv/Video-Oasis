export PYTHONPATH="../..${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run center-frame --model videoauto_r1_qwen3

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run center-frame --model qwen3_vl

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run center-frame --model eagle25
