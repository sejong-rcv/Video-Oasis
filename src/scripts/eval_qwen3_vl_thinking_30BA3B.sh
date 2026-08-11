# setting configuration
model_path=../data/models/Qwen3-VL-30B-A3B-Thinking
output_path=./experiments/Qwen3-VL-30B-A3B-Thinking/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=vqa_total
max_frames=128

accelerate launch --num_processes=1 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model qwen3_vl \
    --model_args pretrained=$model_path,max_frames=$max_frames \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}"