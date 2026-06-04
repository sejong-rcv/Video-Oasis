# setting configuration
model_path=../../data/models/LLaVA-Video-7B-Qwen2
output_path=./experiments/LLaVA-Video-7B-Qwen2/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=v_oasis
max_frames=64

accelerate launch --num_processes=1 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model llava_vid \
    --model_args pretrained=$model_path,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=$max_frames,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after, \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/maxf${max_frames}/"
