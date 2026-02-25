#!/bin/bash
#SBATCH --job-name=demo_sim
#SBATCH --output=../../logs/demo_sim_%j.out
#SBATCH --error=../../logs/demo_sim_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans


# =============================================================================
# GSS Demographic Simulation — SLURM Submission Script
# =============================================================================
#
# Usage:
#   sbatch run_demo_sim.sbatch
#   sbatch run_demo_sim.sbatch --export=MODEL_PATH=/path/to/model,MODEL_NAME=MyModel
#
# Environment variables (optional):
#   MODEL_PATH  - Path to model (default: Meta-Llama-3.1-8B-Instruct)
#   MODEL_NAME  - Model name for output files (default: Llama-3.1-8B-Instruct)
#   ATTN_IMPL   - Attention implementation: sdpa, eager (default: sdpa)
#   BATCH_SIZE  - Batch size (default: 128)
#   MAX_LENGTH  - Max sequence length (default: 512)
#   INCLUDE_LIFESTYLE - Set to "1" to include lifestyle variables in persona
#
# =============================================================================

# Create logs directory
mkdir -p ../../logs

# Print job info
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started at: $(date)"
echo "=============================================="

# Change to working directory
cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/pipelines/demographic_simulation
module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

# Default values (can be overridden via --export)
MODEL_PATH=${MODEL_PATH:-"/project/jevans/maxzhuyt/models/Meta-Llama-3.1-8B-Instruct"}
MODEL_NAME=${MODEL_NAME:-"Llama-3.1-8B-Instruct"}
ATTN_IMPL=${ATTN_IMPL:-"sdpa"}
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_LENGTH=${MAX_LENGTH:-512}
INCLUDE_LIFESTYLE=${INCLUDE_LIFESTYLE:-"0"}

# Build command
CMD="python run_demo_sim.py \
    --model-path ${MODEL_PATH} \
    --model-name ${MODEL_NAME} \
    --attn-impl ${ATTN_IMPL} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --pca-dims 15 \
    --output-dir ../../results"

# Add include-lifestyle flag if requested
if [ "$INCLUDE_LIFESTYLE" = "1" ]; then
    CMD="$CMD --include-lifestyle"
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  ATTN_IMPL:  ${ATTN_IMPL}"
echo "  BATCH_SIZE: ${BATCH_SIZE}"
echo "  MAX_LENGTH: ${MAX_LENGTH}"
echo "  INCLUDE_LIFESTYLE: ${INCLUDE_LIFESTYLE}"
echo ""
echo "Command: ${CMD}"
echo ""

# Run
${CMD}

# Print completion info
echo ""
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
