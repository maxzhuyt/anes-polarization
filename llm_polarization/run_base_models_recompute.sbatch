#!/bin/bash
#SBATCH --job-name=recompute_base
#SBATCH --output=logs/recompute_base_%j.out
#SBATCH --error=logs/recompute_base_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd

# =============================================================================
# Recompute Base Models with Updated Public Issues Prompt
# =============================================================================
#
# Reruns all base models with the new prompt:
#   Public: "{name} makes a statement on {topic}:"
#
# This will create new per-model checkpoints that the explore notebook
# will automatically pick up (most recent files).
#
# Usage:
#   sbatch run_base_models_recompute.sbatch
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

# Run all base models only with updated public issues prompt
CMD="python run_model_comparison.py \
    --families SmolLM3-3B Qwen3-4B Qwen2.5-7B Llama-3.1-8B Gemma-2-9b \
    --model-types base \
    --max-length 128 \
    --pca-dims 5 10 15 \
    --output-dir results"

echo "Configuration:"
echo "  Running: All base models"
echo "  Updated prompt: '{name} makes a statement on {topic}:'"
echo "  Models: SmolLM3-3B, Qwen3-4B, Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9b"
echo ""
echo "Command: ${CMD}"
echo ""

# Temporary: manually filter to base only by modifying the script inline
# We'll run the full script but it will generate new base model checkpoints

${CMD}

echo ""
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
