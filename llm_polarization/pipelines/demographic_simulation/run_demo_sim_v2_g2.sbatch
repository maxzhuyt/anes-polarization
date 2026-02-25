#!/bin/bash
#SBATCH --job-name=demo_v2_g2
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=../../logs/exp0ab_demo_v2_g2_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/pipelines/demographic_simulation

python run_demo_sim_v2.py --group 2

echo "Finished at: $(date)"
