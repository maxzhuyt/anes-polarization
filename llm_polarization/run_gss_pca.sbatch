#!/bin/bash
#SBATCH --job-name=gss_pca
#SBATCH --output=logs/gss_pca_%j.out
#SBATCH --error=logs/gss_pca_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans


# =============================================================================
# GSS Polarization Analysis with PCA - SLURM Submission Script
# =============================================================================
#
# Usage:
#   sbatch run_gss_pca.sbatch
#   sbatch run_gss_pca.sbatch --export=MODEL_PATH=/path/to/model,MODEL_NAME=MyModel
#
# Environment variables (optional):
#   MODEL_PATH  - Path to model (default: Meta-Llama-3.1-8B-Instruct)
#   MODEL_NAME  - Model name for output files (default: Llama-3.1-8B)
#   PCA_DIMS    - PCA dimensions to test (default: "5 10 15 30")
#   SKIP_BOOTSTRAP - Set to "1" to skip bootstrap analysis
#
# =============================================================================

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=============================================="

# Change to working directory
cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization
module load python cuda
source ~/.bashrc
# Activate conda environment (adjust as needed)
conda activate /project/jevans/maxzhuyt/honest_llama_env

# Default values (can be overridden via --export)
MODEL_PATH=${MODEL_PATH:-"/project/jevans/maxzhuyt/models/Qwen1.5-7B-Chat"}
MODEL_NAME=${MODEL_NAME:-"Qwen1.5-7B-Chat"}
PCA_DIMS=${PCA_DIMS:-"5 10 15 30"}
SKIP_BOOTSTRAP=${SKIP_BOOTSTRAP:-"0"}
POLITICIAN_CSV=${POLITICIAN_CSV:-"/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"}

# Copy politician CSV to local data dir to avoid filesystem issues
mkdir -p data
cp -f "${POLITICIAN_CSV}" data/politicians.csv
LOCAL_POLITICIAN_CSV="data/politicians.csv"

# Build command
CMD="python run_gss_pca.py \
    --model-path ${MODEL_PATH} \
    --model-name ${MODEL_NAME} \
    --politician-csv ${LOCAL_POLITICIAN_CSV} \
    --pca-dims ${PCA_DIMS} \
    --public-topics ../question_lists/public_issues.csv \
    --public-polarization ../data/polarization/public_issues_polarization.csv \
    --private-topics ../question_lists/private_life.csv \
    --private-polarization ../data/polarization/private_life_polarization.csv \
    --output-dir results \
    --batch-size 128 \
    --bootstrap-public 126 \
    --bootstrap-private 70"

# Add skip bootstrap flag if requested
if [ "$SKIP_BOOTSTRAP" = "1" ]; then
    CMD="$CMD --skip-bootstrap"
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  POLITICIAN_CSV: ${LOCAL_POLITICIAN_CSV}"
echo "  PCA_DIMS: ${PCA_DIMS}"
echo "  SKIP_BOOTSTRAP: ${SKIP_BOOTSTRAP}"
echo ""
echo "Command: ${CMD}"
echo ""

# Run
${CMD}

# Print completion info
echo ""
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="
