#!/bin/bash
#SBATCH --job-name=dphn_g3_7b
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=../logs/exp0ab_demo_dolphin_g3_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/scratch/midway3/$USER/torchinductor_$SLURM_JOB_ID
export TRITON_CACHE_DIR=/scratch/midway3/$USER/triton_$SLURM_JOB_ID

echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

mkdir -p logs results

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/dolphin_models

python exp0ab_demo_dolphin.py --group 3

echo "Finished at: $(date)"
