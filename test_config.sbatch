#!/bin/bash
#SBATCH --job-name=test_cfg
#SBATCH --output=logs/test_cfg_%j.out
#SBATCH --error=logs/test_cfg_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans

# =============================================================================
# Fast Configuration Testing — SLURM Submission Script
# =============================================================================
#
# Usage:
#   sbatch test_config.sbatch
#   DEMO_FIELDS=core PROMPT_FMT=A sbatch test_config.sbatch
#   DEMO_FIELDS=expanded INCLUDE_LIFESTYLE=1 sbatch test_config.sbatch
#   SAMPLE_FRAC=0.05 sbatch test_config.sbatch
#
# =============================================================================

mkdir -p logs

echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "=============================================="

cd /project/jevans/maxzhuyt/claude_polarization/anes-polarization
module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

# Default values
MODEL_PATH=${MODEL_PATH:-"/project/jevans/maxzhuyt/models/Meta-Llama-3.1-8B-Instruct"}
MODEL_NAME=${MODEL_NAME:-"Llama-3.1-8B-Instruct"}
ATTN_IMPL=${ATTN_IMPL:-"sdpa"}
DEMO_FIELDS=${DEMO_FIELDS:-"all"}
PROMPT_FMT=${PROMPT_FMT:-"B"}
SAMPLE_FRAC=${SAMPLE_FRAC:-"0.01"}
BATCH_SIZE=${BATCH_SIZE:-"100"}
MAX_LENGTH=${MAX_LENGTH:-"512"}
INCLUDE_LIFESTYLE=${INCLUDE_LIFESTYLE:-"0"}

# Build command
CMD="python test_config.py \
    --model-path ${MODEL_PATH} \
    --model-name ${MODEL_NAME} \
    --attn-impl ${ATTN_IMPL} \
    --demo-fields ${DEMO_FIELDS} \
    --prompt-fmt ${PROMPT_FMT} \
    --sample-frac ${SAMPLE_FRAC} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --pca-dims 15 \
    --output-dir llm_results"

if [ "$INCLUDE_LIFESTYLE" = "1" ]; then
    CMD="$CMD --include-lifestyle"
fi

echo ""
echo "Configuration:"
echo "  MODEL:       ${MODEL_NAME}"
echo "  DEMO_FIELDS: ${DEMO_FIELDS}"
echo "  PROMPT_FMT:  ${PROMPT_FMT}"
echo "  SAMPLE_FRAC: ${SAMPLE_FRAC}"
echo "  BATCH_SIZE:  ${BATCH_SIZE}"
echo "  LIFESTYLE:   ${INCLUDE_LIFESTYLE}"
echo ""
echo "Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
