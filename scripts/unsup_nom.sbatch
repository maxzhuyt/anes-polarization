#!/bin/bash

#SBATCH --job-name=unsupervised_nominate
#SBATCH --output=unsupervised_nominate.out
#SBATCH --error=unsupervised_nominate.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
module load python cuda     
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

python3 unsupervised_nominate.py --nominate_csv /project/jevans/maxzhuyt/osfstorage-linear-probe/data/HS116_members.csv \
  --snapshot_root /project/jevans/maxzhuyt/models \