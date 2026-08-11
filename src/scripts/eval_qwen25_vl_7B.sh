# experiment configuration
model_path=../data/models/Qwen2.5-VL-7B-Instruct
output_path=./experiments/Qwen2.5-VL-7B-Instruct/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=vqa_total
max_frames=128

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model qwen2_5_vl \
    --model_args pretrained=$model_path,max_frames=$max_frames \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}"