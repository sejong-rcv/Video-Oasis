#!/bin/bash
# ============================================================
# NoSense QA Ours - Run All Experiments
#
# Models: CLIP, SigLIP, SigLIP2, EVA-CLIP-8B
# Aggregation: mean, max
# Top-K: 32, 128
# ============================================================

set -e

# Default values
GPUS="0,1,2,3,4,5,6,7" 
DATA_PATH="/mnt/users/gtlim/workspace/src/lmms_eval/vqa_total_nonlocal.json" 
FEATURES_DIR="/mnt/gtlim_data/users/gtlim/features" 
OUTPUT_DIR="./json" 
MODELS="siglip-l"
TOP_KS="32,128"
AGGREGATIONS="mean,max"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --features_dir)
            FEATURES_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --top_ks)
            TOP_KS="$2"
            shift 2
            ;;
        --aggregations)
            AGGREGATIONS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus          Comma-separated GPU IDs (default: 0)"
            echo "  --data_path     Path to benchmark JSON"
            echo "  --features_dir  Base directory for features"
            echo "  --output_dir    Output directory for results"
            echo "  --models        Comma-separated model names"
            echo "  --top_ks        Comma-separated top-k values (default: 32,128)"
            echo "  --aggregations  Comma-separated aggregation methods (default: mean,max)"
            echo ""
            echo "Example:"
            echo "  $0 --gpus 0,1,2,3 --models siglip2-giant,eva-clip-8b"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "NoSense QA Ours - Experiment Runner"
echo "============================================================"
echo "GPUs: $GPUS"
echo "Models: $MODELS"
echo "Top-K: $TOP_KS"
echo "Aggregations: $AGGREGATIONS"
echo "Data: $DATA_PATH"
echo "Features: $FEATURES_DIR"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# # Change to project directory
# cd "$(dirname "$0")/.."

# Run all experiments
python nosense_qa_ours.py \
    --run_all \
    --gpus "$GPUS" \
    --data_path "$DATA_PATH" \
    --features_dir "$FEATURES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODELS" \
    --top_ks "$TOP_KS" \
    --aggregations "$AGGREGATIONS"

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
