#!/bin/bash
#SBATCH --job-name=dphn_small
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/exp0ab_demo_dolphin_small_%j.out

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

# Groups 2 3 4 5: llama-8b, mistral-7b, qwen-3b, gemma2-2b
# GSS data and topics are loaded once; models run sequentially.
python exp0ab_demo_dolphin.py --groups 2 3 4 5

echo "Finished at: $(date)"
