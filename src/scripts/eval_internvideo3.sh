# setting configuration
model_path=../data/models/InternVideo3-8B-Instruct
output_path=./experiments/InternVideo3-8B-Instruct/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=v_oasis
max_frames=128

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model internvideo3 \
    --model_args pretrained=$model_path,max_frames_num=$max_frames \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/maxf${max_frames}/"
