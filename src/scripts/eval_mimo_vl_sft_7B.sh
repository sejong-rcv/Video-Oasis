# token configuration
video_min_tokens=16
video_max_tokens=768
video_total_tokens=128000
video_min_pixels=$((video_min_tokens * 32 * 32))
video_max_pixels=$((video_max_tokens * 32 * 32))
video_total_pixels=$((video_total_tokens * 32 * 32))

# setting configuration
model_path=../data/models/MiMo-VL-7B-SFT-2508
output_path=./experiments//MiMo-VL-7B-SFT-2508/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=v_oasis
max_frames=128

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model mimo_vl \
    --model_args pretrained=$model_path,video_min_pixels=$video_min_pixels,video_max_pixels=$video_max_pixels,video_total_pixels=$video_total_pixels,max_frames=$max_frames \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/min${video_min_tokens}_max${video_max_tokens}_total${video_total_tokens}_maxf${max_frames}/"
