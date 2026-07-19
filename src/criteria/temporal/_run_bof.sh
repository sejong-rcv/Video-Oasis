export PYTHONPATH="../..${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run bag-of-frames --model clip-vit-l-14

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run bag-of-frames --model eva-clip-8b

CUDA_VISIBLE_DEVICES=0 python -m criteria.temporal.run bag-of-frames --model longclip
