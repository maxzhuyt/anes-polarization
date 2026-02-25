#!/bin/bash

#SBATCH --job-name=logit_pc1
#SBATCH --output=logit_pc1.out
#SBATCH --error=logit_pc1.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
module load python cuda     
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

python3 logit_pc1.py --nominate_csv /project/jevans/maxzhuyt/osfstorage-linear-probe/data/HS116_members.csv \
  --snapshot_root /project/jevans/maxzhuyt/models \
  --model_id google/gemma-2-9b-it \
  --batch_size 64 \
