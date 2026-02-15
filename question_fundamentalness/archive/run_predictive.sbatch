#!/bin/bash
#SBATCH --job-name=gss-predictive
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gss_predictive_%j.out
#SBATCH --error=logs/gss_predictive_%j.err

# GSS Polarization - Predictive Power Analysis Only
# This is the slowest analysis, can run overnight

echo "=========================================="
echo "GSS Predictive Power Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"
echo "=========================================="

mkdir -p logs

cd /project/jevans/maxzhuyt/gss-polarization
source /project/jevans/maxzhuyt/honest_llama_env/bin/activate

N_JOBS=${SLURM_CPUS_PER_TASK:-14}
OUTPUT_DIR="results"
MIN_YEARS=2

echo "N_JOBS: $N_JOBS"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

mkdir -p $OUTPUT_DIR

python scripts/run_predictive_power.py \
    --n_jobs $N_JOBS \
    --output_prefix $OUTPUT_DIR/predictive \
    --min_years $MIN_YEARS

echo ""
echo "=========================================="
echo "Predictive Power Analysis Complete!"
echo "Finished: $(date)"
echo "=========================================="
