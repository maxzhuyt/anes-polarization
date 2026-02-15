#!/bin/bash
#SBATCH --job-name=overnight_ssd
#SBATCH --output=logs/ssd_gpu_%j.out
#SBATCH --error=logs/ssd_gpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd

# =============================================================================
# Overnight Experiments on ssd-gpu (A100 40GB)
# =============================================================================
#
# Runs polarization mechanism experiments:
#   - Experiment 3: Cross-topic activation coherence (issue bundling)
#   - Experiment 4: Moral foundations intensity effects
#   - Experiment 5: Elite amplification ratio
#   - Experiment 6: Overlap coefficient (false polarization)
#   - Experiment 8: Affective polarization (affect vs ideology)
#
# Total estimated runtime: ~8 hours (4-hour buffer)
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

# Set PyTorch memory config
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo ""
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# =============================================================================
# Experiment 3: Cross-Topic Coherence
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 3: CROSS-TOPIC COHERENCE"
echo "Started: $(date)"
echo "=============================================="
echo ""

if [ -f exp3_coherence.py ]; then
    python exp3_coherence.py
    EXIT_CODE_EXP3=$?

    if [ $EXIT_CODE_EXP3 -ne 0 ]; then
        echo "ERROR: Experiment 3 failed with exit code $EXIT_CODE_EXP3"
    else
        echo "Experiment 3 completed successfully"
    fi
else
    echo "SKIP: exp3_coherence.py not found"
    EXIT_CODE_EXP3=0
fi

echo ""
echo "Experiment 3 finished at: $(date)"
echo ""

# =============================================================================
# Experiment 4: Moral Foundations
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 4: MORAL FOUNDATIONS"
echo "Started: $(date)"
echo "=============================================="
echo ""

if [ -f exp4_moral_foundations.py ]; then
    python exp4_moral_foundations.py
    EXIT_CODE_EXP4=$?

    if [ $EXIT_CODE_EXP4 -ne 0 ]; then
        echo "ERROR: Experiment 4 failed with exit code $EXIT_CODE_EXP4"
    else
        echo "Experiment 4 completed successfully"
    fi
else
    echo "SKIP: exp4_moral_foundations.py not found"
    EXIT_CODE_EXP4=0
fi

echo ""
echo "Experiment 4 finished at: $(date)"
echo ""

# =============================================================================
# Experiment 5: Elite Amplification
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 5: ELITE AMPLIFICATION"
echo "Started: $(date)"
echo "=============================================="
echo ""

if [ -f exp5_elite_amplification.py ]; then
    python exp5_elite_amplification.py
    EXIT_CODE_EXP5=$?

    if [ $EXIT_CODE_EXP5 -ne 0 ]; then
        echo "ERROR: Experiment 5 failed with exit code $EXIT_CODE_EXP5"
    else
        echo "Experiment 5 completed successfully"
    fi
else
    echo "SKIP: exp5_elite_amplification.py not found"
    EXIT_CODE_EXP5=0
fi

echo ""
echo "Experiment 5 finished at: $(date)"
echo ""

# =============================================================================
# Experiment 6: False Polarization (Overlap)
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 6: FALSE POLARIZATION"
echo "Started: $(date)"
echo "=============================================="
echo ""

if [ -f exp6_false_polarization.py ]; then
    python exp6_false_polarization.py
    EXIT_CODE_EXP6=$?

    if [ $EXIT_CODE_EXP6 -ne 0 ]; then
        echo "ERROR: Experiment 6 failed with exit code $EXIT_CODE_EXP6"
    else
        echo "Experiment 6 completed successfully"
    fi
else
    echo "SKIP: exp6_false_polarization.py not found"
    EXIT_CODE_EXP6=0
fi

echo ""
echo "Experiment 6 finished at: $(date)"
echo ""

# =============================================================================
# Experiment 8: Affective Polarization
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT 8: AFFECTIVE POLARIZATION"
echo "Started: $(date)"
echo "=============================================="
echo ""

if [ -f exp8_affective_polarization.py ]; then
    python exp8_affective_polarization.py
    EXIT_CODE_EXP8=$?

    if [ $EXIT_CODE_EXP8 -ne 0 ]; then
        echo "ERROR: Experiment 8 failed with exit code $EXIT_CODE_EXP8"
    else
        echo "Experiment 8 completed successfully"
    fi
else
    echo "SKIP: exp8_affective_polarization.py not found"
    EXIT_CODE_EXP8=0
fi

echo ""
echo "Experiment 8 finished at: $(date)"
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
echo "  Experiment 3: $EXIT_CODE_EXP3"
echo "  Experiment 4: $EXIT_CODE_EXP4"
echo "  Experiment 5: $EXIT_CODE_EXP5"
echo "  Experiment 6: $EXIT_CODE_EXP6"
echo "  Experiment 8: $EXIT_CODE_EXP8"
echo ""

# Exit with error if any experiment failed
if [ $EXIT_CODE_EXP3 -ne 0 ] || [ $EXIT_CODE_EXP4 -ne 0 ] || [ $EXIT_CODE_EXP5 -ne 0 ] || [ $EXIT_CODE_EXP6 -ne 0 ] || [ $EXIT_CODE_EXP8 -ne 0 ]; then
    echo "At least one experiment failed!"
    exit 1
else
    echo "All experiments succeeded!"
    exit 0
fi
