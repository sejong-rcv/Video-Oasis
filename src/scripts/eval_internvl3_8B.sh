# setting configuration
model_path=../data/models/InternVL3-8B
output_path=./experiments/InternVL3-8B/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=v_oasis
max_frames=128

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model internvl3 \
    --model_args pretrained=$model_path,num_frame=$max_frames \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/maxf${max_frames}/"