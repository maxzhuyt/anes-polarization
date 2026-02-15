#!/bin/bash
#SBATCH --job-name=model_cmp
#SBATCH --output=logs/model_cmp_%j.out
#SBATCH --error=logs/model_cmp_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans

# =============================================================================
# Model Comparison: Base vs Instruct vs Reasoning — SLURM Submission Script
# =============================================================================
#
# Usage:
#   sbatch run_model_comparison.sbatch
#   FAMILIES="Qwen3-4B" sbatch run_model_comparison.sbatch
#   FAMILIES="Qwen3-4B Llama-3.1-8B" sbatch run_model_comparison.sbatch
#
# =============================================================================

mkdir -p logs

echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "=============================================="

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization
module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

# Default values
BATCH_SIZE=${BATCH_SIZE:-"80"}
MAX_LENGTH=${MAX_LENGTH:-"128"}
PCA_DIMS=${PCA_DIMS:-"5 10 15"}

# Build command
CMD="python run_model_comparison.py \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --pca-dims ${PCA_DIMS} \
    --output-dir results"

# Optional: subset of families
if [ -n "$FAMILIES" ]; then
    CMD="$CMD --families ${FAMILIES}"
fi

# Optional: skip plot generation
if [ "$SKIP_PLOTS" = "1" ]; then
    CMD="$CMD --skip-plots"
fi

echo ""
echo "Configuration:"
echo "  FAMILIES:   ${FAMILIES:-all}"
echo "  BATCH_SIZE: ${BATCH_SIZE}"
echo "  MAX_LENGTH: ${MAX_LENGTH}"
echo "  PCA_DIMS:   ${PCA_DIMS}"
echo "  SKIP_PLOTS: ${SKIP_PLOTS:-0}"
echo ""
echo "Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
