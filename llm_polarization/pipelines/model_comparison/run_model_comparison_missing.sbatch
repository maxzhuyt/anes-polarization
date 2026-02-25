#!/bin/bash
#SBATCH --job-name=model_cmp_missing
#SBATCH --output=logs/model_cmp_missing_%j.out
#SBATCH --error=logs/model_cmp_missing_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd

# =============================================================================
# Model Comparison: Missing Models (Ministral-3-8B + Gemma-2-9b)
# =============================================================================
#
# This script completes the model comparison by running the 5 models that
# failed due to Ministral-3 transformers compatibility issue.
#
# Usage:
#   sbatch run_model_comparison_missing.sbatch
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

# Upgrade transformers for Ministral-3 support
echo ""
echo "Upgrading transformers for Ministral-3 support..."
pip install --upgrade 'transformers>=4.48.0' --quiet
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

# Run only Gemma (Ministral-3 is multimodal, not compatible)
CMD="python run_model_comparison.py \
    --families Gemma-2-9b \
    --batch-size 80 \
    --max-length 128 \
    --pca-dims 5 10 15 \
    --output-dir results"

echo "Configuration:"
echo "  FAMILIES:   Gemma-2-9b only (Ministral-3 is multimodal, skipped)"
echo "  BATCH_SIZE: per-model (120 for Gemma)"
echo "  MAX_LENGTH: 128"
echo "  PCA_DIMS:   5 10 15"
echo ""
echo "Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
