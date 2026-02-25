#!/bin/bash
#SBATCH --job-name=dphn_resume345
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/exp0ab_demo_dolphin_resume_g345_%j.out

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/scratch/midway3/$USER/torchinductor_$SLURM_JOB_ID
export TRITON_CACHE_DIR=/scratch/midway3/$USER/triton_$SLURM_JOB_ID

echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

mkdir -p logs results

cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments/dolphin_models

# Resume groups 3 (dphn-mistral7b), 4 (dphn-qwen3b), 5 (dphn-gemma2b)
# --resume skips (model, category, condition) combos that already have checkpoints
# Group 3 completed: public/rhetorical + public/stance → resumes at public/survey + all private
python exp0ab_demo_dolphin.py --groups 3 4 5 --resume

echo "Finished at: $(date)"
