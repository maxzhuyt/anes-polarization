#!/bin/bash

#SBATCH --job-name=sanity_softmax
#SBATCH --output=sanity_softmax.out
#SBATCH --error=sanity_softmax.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
module load python cuda     
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

python3 sanity_softmax.py --nominate_csv /project/jevans/maxzhuyt/osfstorage-linear-probe/data/HS116_members.csv \
  --snapshot_root /project/jevans/maxzhuyt/models \
  --model_id google/gemma-2-9b-it \
