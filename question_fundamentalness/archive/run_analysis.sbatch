#!/bin/bash
#SBATCH --job-name=gss-analysis
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gss_analysis_%j.out
#SBATCH --error=logs/gss_analysis_%j.err

# GSS Polarization - Full Analysis Pipeline
# Runs all 5 analysis methods with multiprocessing

echo "=========================================="
echo "GSS Question Hierarchy Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Change to project directory
cd /project/jevans/maxzhuyt/gss-polarization

# Activate virtual environment
source /project/jevans/maxzhuyt/honest_llama_env/bin/activate

# Configuration
N_JOBS=${SLURM_CPUS_PER_TASK:-14}
OUTPUT_DIR="results"
MIN_YEARS=2

echo ""
echo "Configuration:"
echo "  N_JOBS: $N_JOBS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  MIN_YEARS: $MIN_YEARS"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run all analyses
python scripts/run_all_analyses.py \
    --n_jobs $N_JOBS \
    --output_dir $OUTPUT_DIR \
    --min_years $MIN_YEARS

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "Finished: $(date)"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/
