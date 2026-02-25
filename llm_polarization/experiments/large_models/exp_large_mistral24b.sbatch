#!/bin/bash
#SBATCH --job-name=lg_mistral24b
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/exp_large_mistral24b_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/large_models

python exp_large_models.py --model mistral-24b

echo "Finished at: $(date)"
