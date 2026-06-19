CUDA_VISIBLE_DEVICES=0 python pred_mllm.py --model_version qwen25_vl
CUDA_VISIBLE_DEVICES=1 python pred_mllm.py --model_version qwen3_vl
CUDA_VISIBLE_DEVICES=2 python pred_mllm.py --model_version eagle25
