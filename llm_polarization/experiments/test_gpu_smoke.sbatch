#!/bin/bash
#SBATCH --job-name=gpu_smoke
#SBATCH --output=logs/gpu_smoke_%j.out
#SBATCH --error=logs/gpu_smoke_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans

mkdir -p logs

echo "GPU Smoke Test - Job $SLURM_JOB_ID on $(hostname) at $(date)"

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python test_gpu_smoke.py
EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"

exit $EXIT_CODE
