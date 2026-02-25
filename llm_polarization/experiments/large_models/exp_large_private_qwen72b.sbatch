#!/bin/bash
#SBATCH --job-name=priv_qw72b
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=../logs/exp_large_private_qwen72b_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/large_models

# qwen2.5-72b on 2x H100 (160GB) — 72B BF16 ~144GB requires 2 GPUs
python exp_large_models.py --model qwen2.5-72b --topic-set private

echo "Finished at: $(date)"
