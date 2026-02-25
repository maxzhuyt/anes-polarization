#!/bin/bash
#SBATCH --job-name=priv_yi34b
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/exp_large_private_yi34b_dolphin_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/large_models

# dolphin-34b (yi-34b-dolphin) on 1x H100 (80GB) — 34B BF16 ~68GB fits on single H100
python exp_large_models.py --model dolphin-34b --topic-set private

echo "Finished at: $(date)"
