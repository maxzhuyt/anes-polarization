#!/bin/bash

#SBATCH --job-name=manifesto
#SBATCH --output=manifesto.out
#SBATCH --error=manifesto.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
module load python cuda     
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

python3 manifesto.py \
  --csv /project/jevans/maxzhuyt/data/latest_manifesto_2024enriched.csv \
  --snapshot_root /project/jevans/maxzhuyt/models \
  --batch_size 6 --max_length 512 \
  --gpus 1
