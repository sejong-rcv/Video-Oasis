# token configuration
video_min_tokens=16
video_max_tokens=768
video_total_tokens=16384
video_min_pixels=$((video_min_tokens * 28 * 28))
video_max_pixels=$((video_max_tokens * 28 * 28))
video_total_pixels=$((video_total_tokens * 28 * 28))

# setting configuration
model_path=../../data/models/Eagle2.5-8B
output_path=./experiments/Eagle2.5-8B_16K/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=v_oasis
max_frames=128

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model eagle2_5 \
    --model_args pretrained=$model_path,video_min_pixels=$video_min_pixels,video_max_pixels=$video_max_pixels,video_total_pixels=$video_total_pixels,max_frames=$max_frames,image_min_pixels=$image_min_pixels,image_max_pixels=$image_max_pixels \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/maxf${max_frames}/"