CUDA_VISIBLE_DEVICES=5 python3 test.py --model_version llama
CUDA_VISIBLE_DEVICES=5 python3 test.py --model_version qwen
CUDA_VISIBLE_DEVICES=6 python3 test.py --model_version mistral
CUDA_VISIBLE_DEVICES=6 python3 test.py --model_version qwen25_vl
CUDA_VISIBLE_DEVICES=7 python3 test.py --model_version qwen3_vl
CUDA_VISIBLE_DEVICES=7 python3 test.py --model_version eagle25