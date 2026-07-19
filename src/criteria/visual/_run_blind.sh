export PYTHONPATH="../..${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python -m criteria.visual.run blind --model qwen25_vl

CUDA_VISIBLE_DEVICES=0 python -m criteria.visual.run blind --model qwen3_vl

CUDA_VISIBLE_DEVICES=0 python -m criteria.visual.run blind --model eagle25
