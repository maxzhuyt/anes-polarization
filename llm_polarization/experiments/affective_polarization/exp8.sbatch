#!/bin/bash
#SBATCH --job-name=exp8_affective
#SBATCH --output=../logs/exp8_%j.out
#SBATCH --error=../logs/exp8_%j.err
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd

mkdir -p logs

echo "=============================================="
echo "EXPERIMENT 8: AFFECTIVE VS POLICY POLARIZATION (REDESIGNED)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "=============================================="

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/affective_polarization

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python exp8_affective_polarization.py

echo ""
echo "Finished at: $(date)"
