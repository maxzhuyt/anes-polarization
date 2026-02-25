#!/bin/bash
#SBATCH --job-name=priv_glm_air
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/exp_large_private_glm_air_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/large_models

# glm-air on 4x H100 (320GB) — very large model requires all 4 GPUs
python exp_large_models.py --model glm-air --topic-set private

echo "Finished at: $(date)"
