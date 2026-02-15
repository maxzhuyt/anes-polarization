#!/bin/bash
#SBATCH --job-name=test_prompts
#SBATCH --output=logs/test_prompts_%j.out
#SBATCH --error=logs/test_prompts_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd

# =============================================================================
# Base Model Prompt Sensitivity Test
# =============================================================================
#
# Tests whether base models are sensitive to prompt formatting by comparing:
#   Original:    "{name} shares views on {topic}, saying"
#   Alternative: "{name} makes statement on {topic}:"
#
# Runs 20 topics per category across all 6 base models.
#
# Usage:
#   sbatch test_base_prompts.sbatch
#   N_TOPICS=30 sbatch test_base_prompts.sbatch
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

# Upgrade transformers for Ministral-3 support (if needed)
echo ""
echo "Checking transformers version..."
pip install --upgrade 'transformers>=4.48.0' --quiet
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

# Default values
N_TOPICS=${N_TOPICS:-"20"}
MAX_LENGTH=${MAX_LENGTH:-"128"}

# Build command
CMD="python test_base_prompts.py \
    --n-topics ${N_TOPICS} \
    --max-length ${MAX_LENGTH} \
    --pca-dims 5 10 15 \
    --output-dir results \
    --seed 42"

echo "Configuration:"
echo "  Base models: 5 (SmolLM3, Qwen3-4B, Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9b)"
echo "  Topics per category: ${N_TOPICS}"
echo "  MAX_LENGTH: ${MAX_LENGTH}"
echo ""
echo "Command: ${CMD}"
echo ""

${CMD}

echo ""
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
