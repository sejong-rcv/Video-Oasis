CUDA_VISIBLE_DEVICES=0 python3 test.py --model_version llama --summary_file caption/total_summary.json
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_version qwen --summary_file caption/total_summary.json
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_version mistral --summary_file caption/total_summary.json
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_version qwen25_vl --summary_file caption/total_summary.json
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_version qwen3_vl --summary_file caption/total_summary.json
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_version eagle25 --summary_file caption/total_summary.json