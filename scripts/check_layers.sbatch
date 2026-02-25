#!/bin/bash

#SBATCH --job-name=check_layers
#SBATCH --output=check_layers.out
#SBATCH --error=check_layers.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
module load python cuda     
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

python3 check_layers.py