#!/bin/bash
#SBATCH --job-name=overnight_jevans
#SBATCH --output=logs/jevans_gpu_%j.out
#SBATCH --error=logs/jevans_gpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans

# =============================================================================
# Overnight Experiments on jevans-gpu (H100 80GB)
# =============================================================================
#
# Runs model internal experiments:
#   - Experiment 1:  Encoding strength (base vs instruct vs reasoning)
#   - Experiment 1R: Replication with different topic sample
#   - Experiment 2:  Layer-wise encoding depth
#   - Experiment 7:  Attention head causal contributions
#
# Total estimated runtime: ~10 hours (2-hour buffer)
#
# =============================================================================

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "=============================================="

# Navigate to experiments directory
cd /project/jevans/maxzhuyt/gss_polarization/llm_polarization/experiments

# Load modules
module load python cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

# Set PyTorch memory config (prevent fragmentation)
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo ""
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# =============================================================================
# Experiment 1: Encoding Strength
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 1: ENCODING STRENGTH"
echo "Started: $(date)"
echo "=============================================="
echo ""

python exp1_encoding_strength.py

EXIT_CODE_EXP1=$?

if [ $EXIT_CODE_EXP1 -ne 0 ]; then
    echo "ERROR: Experiment 1 failed with exit code $EXIT_CODE_EXP1"
else
    echo "Experiment 1 completed successfully"
fi

echo ""
echo "Experiment 1 finished at: $(date)"
echo ""

# =============================================================================
# Experiment 1R: Replication (different topic sample)
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 1R: REPLICATION"
echo "Started: $(date)"
echo "=============================================="
echo ""

python exp1_encoding_strength.py --replication

EXIT_CODE_EXP1R=$?

if [ $EXIT_CODE_EXP1R -ne 0 ]; then
    echo "ERROR: Experiment 1R failed with exit code $EXIT_CODE_EXP1R"
else
    echo "Experiment 1R completed successfully"
fi

echo ""
echo "Experiment 1R finished at: $(date)"
echo ""

# =============================================================================
# Experiment 2: Layer-Wise Encoding Depth
# =============================================================================

# NOTE: Experiment 2 script not yet implemented
# Placeholder for now

echo ""
echo "=============================================="
echo "EXPERIMENT 2: LAYER-WISE DEPTH"
echo "Started: $(date)"
echo "=============================================="
echo ""

if [ -f exp2_layer_depth.py ]; then
    python exp2_layer_depth.py
    EXIT_CODE_EXP2=$?

    if [ $EXIT_CODE_EXP2 -ne 0 ]; then
        echo "ERROR: Experiment 2 failed with exit code $EXIT_CODE_EXP2"
    else
        echo "Experiment 2 completed successfully"
    fi
else
    echo "SKIP: exp2_layer_depth.py not found"
    EXIT_CODE_EXP2=0
fi

echo ""
echo "Experiment 2 finished at: $(date)"
echo ""

# =============================================================================
# Experiment 7: Attention Head Contributions
# =============================================================================

# NOTE: Experiment 7 script not yet implemented
# Placeholder for now

echo ""
echo "=============================================="
echo "EXPERIMENT 7: ATTENTION FLOW"
echo "Started: $(date)"
echo "=============================================="
echo ""

if [ -f exp7_attention_flow.py ]; then
    python exp7_attention_flow.py
    EXIT_CODE_EXP7=$?

    if [ $EXIT_CODE_EXP7 -ne 0 ]; then
        echo "ERROR: Experiment 7 failed with exit code $EXIT_CODE_EXP7"
    else
        echo "Experiment 7 completed successfully"
    fi
else
    echo "SKIP: exp7_attention_flow.py not found"
    EXIT_CODE_EXP7=0
fi

echo ""
echo "Experiment 7 finished at: $(date)"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished at: $(date)"
echo "=============================================="
echo ""
echo "Exit Codes:"
echo "  Experiment 1:  $EXIT_CODE_EXP1"
echo "  Experiment 1R: $EXIT_CODE_EXP1R"
echo "  Experiment 2:  $EXIT_CODE_EXP2"
echo "  Experiment 7:  $EXIT_CODE_EXP7"
echo ""

# Exit with error if any experiment failed
if [ $EXIT_CODE_EXP1 -ne 0 ] || [ $EXIT_CODE_EXP1R -ne 0 ] || [ $EXIT_CODE_EXP2 -ne 0 ] || [ $EXIT_CODE_EXP7 -ne 0 ]; then
    echo "At least one experiment failed!"
    exit 1
else
    echo "All experiments succeeded!"
    exit 0
fi
