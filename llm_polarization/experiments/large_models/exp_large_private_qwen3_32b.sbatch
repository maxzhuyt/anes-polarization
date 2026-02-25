#!/bin/bash
#SBATCH --job-name=priv_qw3_32b
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/exp_large_private_qwen3_32b_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/large_models

# qwen3-32b on 1x H100 (80GB) — 32B BF16 ~64GB fits on single H100
python exp_large_models.py --model qwen3-32b --topic-set private

echo "Finished at: $(date)"
