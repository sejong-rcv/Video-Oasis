CUDA_VISIBLE_DEVICES=0 python pred_llm.py --model_version qwen
CUDA_VISIBLE_DEVICES=1 python pred_llm.py --model_version mistral
CUDA_VISIBLE_DEVICES=2 python pred_llm.py --model_version llama