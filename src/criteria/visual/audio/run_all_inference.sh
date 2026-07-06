GPUS="0,1,2,3"   # set GPU ID
LOG_DIR="./logs"  

mkdir -p "$LOG_DIR"

MODELS=(
    "qwen25_vl"
    "qwen3_vl"
    "eagle25"
)

DATASETS=(
    "longvideobench"
    "mlvu_test"
    "mmr-v"
    "mvbench"
    "rtv-bench"
    "tvbench"
    "vcr-bench"
    "video-holmes"
    "video-mme"
)

echo "========================================================"
echo "  Start Batch Inference"
echo "   GPUs: $GPUS"
echo "   Models: ${#MODELS[@]} types"
echo "   Datasets: ${#DATASETS[@]} types"
echo "========================================================"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="$LOG_DIR/${MODEL}_${DATASET}_${TIMESTAMP}.log"

        echo ""
        echo "▶  [Running] Model: $MODEL | Dataset: $DATASET"
        echo "    Logs will be saved to: $LOG_FILE"
        
        OMP_NUM_THREADS=1 python run_infer_mp.py \
            --gpus "$GPUS" \
            --model_version "$MODEL" \
            --dataset "$DATASET" \
            2>&1 | tee "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "[Success] $MODEL - $DATASET"
        else
            echo "[Failed] $MODEL - $DATASET (Check log: $LOG_FILE)"
        fi
        
        echo "--------------------------------------------------------"
    done
done