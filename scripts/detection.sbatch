#!/bin/bash

#SBATCH --job-name=detection
#SBATCH --output=detection.out
#SBATCH --error=detection.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python cuda     
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

python3 detection_concept.py