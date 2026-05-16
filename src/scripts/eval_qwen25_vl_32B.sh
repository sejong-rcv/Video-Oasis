# data configuration
image_min_tokens=128
image_max_tokens=16384
video_min_tokens=16
video_max_tokens=768
video_total_tokens=16384

image_min_pixels=$((image_min_tokens * 28 * 28))
image_max_pixels=$((image_max_tokens * 28 * 28))
video_min_pixels=$((video_min_tokens * 28 * 28))
video_max_pixels=$((video_max_tokens * 28 * 28))
video_total_pixels=$((video_total_tokens * 28 * 28))

# experiment configuration
model_path=/mnt/gtlim_data/users/gtlim/models/Qwen2.5-VL-32B-Instruct
output_path=./experiments/qwen_benchmark/Qwen2.5-VL-32B-Instruct/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=vqa_total
max_frames=128

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model qwen2_5_vl \
    --model_args pretrained=$model_path,video_min_pixels=$video_min_pixels,video_max_pixels=$video_max_pixels,video_total_pixels=$video_total_pixels,max_frames=$max_frames,image_min_pixels=$image_min_pixels,image_max_pixels=$image_max_pixels \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/min${video_min_tokens}_max${video_max_tokens}_total${video_total_tokens}_maxf${max_frames}/"
