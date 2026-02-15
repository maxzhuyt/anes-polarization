#!/bin/bash
#SBATCH --job-name=qwen25_comparison
#SBATCH --output=logs/qwen25_cmp_%j.out
#SBATCH --error=logs/qwen25_cmp_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans

mkdir -p logs

echo "=============================================="
echo "Qwen2.5-7B Model Comparison"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "=============================================="

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

export PYTORCH_ALLOC_CONF=expandable_segments:True

python run_model_comparison.py \
    --families Qwen2.5-7B \
    --batch-size 160 \
    --max-length 128 \
    --pca-dims 5 10 15 \
    --output-dir results

echo ""
echo "Finished at: $(date)"
