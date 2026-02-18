#!/bin/bash
#SBATCH --job-name=exp10_resid
#SBATCH --output=logs/exp10_%j.out
#SBATCH --error=logs/exp10_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans

mkdir -p logs

echo "=============================================="
echo "EXPERIMENT 10: RESIDUAL TOPIC-SPECIFIC SIGNAL"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "=============================================="

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python exp10_residual_signal.py

echo ""
echo "Finished at: $(date)"
