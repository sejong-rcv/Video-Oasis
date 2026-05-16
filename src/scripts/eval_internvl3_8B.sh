# experiment configuration
model_path=/mnt/gtlim_data/users/gtlim/models/InternVL3-8B
output_path=./experiments/benchmark_internvl/InternVL3-8B/

# task list
# tasks=(videomme_boxed mvbench_boxed longvideobench_val_v_boxed mmvu_val_mc_boxed video_mmmu_boxed charades_boxed activitynet_tvg_boxed nextgqa_boxed)
# tasks=(videomme_boxed)
tasks=(msrbench)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
for task in "${tasks[@]}"; do
    echo "Running task $task"

    if [[ "$task" == "video_mmmu" || "$task" == "mmvu_val_mc" ]]; then
    max_frames=64
    elif [[ "$task" == "activitynet_tvg" ]]; then
    max_frames=512
    else
    max_frames=128
    fi

    accelerate launch --num_processes=8 --main_process_port=$master_port -m lmms_eval.__main__ \
    --model internvl3 \
    --model_args pretrained=$model_path,num_frame=$max_frames \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --output_path "${output_path}/maxf${max_frames}/"
    sleep 30
done
