#!/bin/bash

# ==========================================
# ⚙️ 설정 (Configuration)
# ==========================================
GPUS="0,1,2,3"   # 사용할 GPU ID
LOG_DIR="./logs"  # 로그 저장할 폴더

# 폴더 없으면 생성
mkdir -p "$LOG_DIR"

# 1. 모델 목록 (Python 코드의 choices와 일치해야 함)
MODELS=(
    "qwen25_vl"
    "qwen3_vl"
    "eagle25"
)

# 2. 데이터셋 목록
DATASETS=(
    "longvideobench"
    "mlvu_test"
    "mmrvbench"
    "mvbench"
    "rtv-bench"
    "tvbench"
    "vcr-bench"
    "video-holmes"
    "video-mme"
)

# ==========================================
# 🚀 실행 루프 (Execution Loop)
# ==========================================

echo "========================================================"
echo "🔥 Start Batch Inference"
echo "   GPUs: $GPUS"
echo "   Models: ${#MODELS[@]} types"
echo "   Datasets: ${#DATASETS[@]} types"
echo "========================================================"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        # 현재 시간 (로그 파일명용)
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="$LOG_DIR/${MODEL}_${DATASET}_${TIMESTAMP}.log"

        echo ""
        echo "▶️  [Running] Model: $MODEL | Dataset: $DATASET"
        echo "    Logs will be saved to: $LOG_FILE"
        
        # --- Python 실행 ---
        # 1. 2>&1 | tee ... : 화면에도 출력하고 로그 파일에도 저장
        # 2. OMP_NUM_THREADS=1 : 멀티프로세싱 부하 방지
        OMP_NUM_THREADS=1 python run_infer_mp.py \
            --gpus "$GPUS" \
            --model_version "$MODEL" \
            --dataset "$DATASET" \
            2>&1 | tee "$LOG_FILE"

        # 실행 결과 확인 (Exit Code)
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ [Success] $MODEL - $DATASET"
        else
            echo "❌ [Failed] $MODEL - $DATASET (Check log: $LOG_FILE)"
            # 실패해도 다음 실험을 위해 멈추지 않고 계속 진행 (원하면 exit 1 추가)
        fi
        
        echo "--------------------------------------------------------"
    done
done

echo "🏆 All batch jobs finished!"