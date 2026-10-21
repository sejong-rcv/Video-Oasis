# setting configuration
model_path=../data/models/gemma-4-12B-it
output_path=./experiments/gemma-4-12B-it/
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# experiment configuration
task=v_oasis
max_frames=64

accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model gemma4 \
    --model_args pretrained=$model_path,max_frames=$max_frames \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}"
