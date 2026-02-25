#!/bin/bash

#SBATCH --job-name=ches
#SBATCH --output=ches.out
#SBATCH --error=ches.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
module load python cuda     
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

python3 ches.py \
  --ches_csv /project/jevans/maxzhuyt/data/CHES_2024_annotated.csv \
  --snapshot_root /project/jevans/maxzhuyt/models \
  --batch_size 6 \
  --gpus 1
