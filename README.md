# LLM Polarization Analysis

Analyzing how LLM activation polarization correlates with survey-measured partisan polarization (GSS and ANES).

## Project Structure

```
├── analysis_gss.ipynb              # Main analysis: GSS single-model
├── analysis_gss_multimodel.ipynb   # Main analysis: GSS multi-model comparison
├── analysis_gss_pca.ipynb          # PCA-based analysis
├── analysis_anes.ipynb             # Supplementary: ANES analysis
├── demographic_simulation.ipynb    # Demographic persona simulation analysis
├── explore_demo_sim.ipynb          # Explore demographic simulation results
├── explore_test_config.ipynb       # Explore test_config results, topic exclusion analysis
├── enrich_gss_variables.ipynb      # Enrich GSS variable metadata
├── test_config.ipynb               # Interactive config testing (superseded by .py)
│
├── config.py                       # Topics, model paths, settings
├── model_utils.py                  # Model loading, activation extraction (SDPA, GQA)
├── metrics_utils.py                # Polarization metrics (Mahalanobis, etc.)
├── prompt_utils.py                 # Prompt generators
├── run_gss_pca.py                  # PCA + Mahalanobis computation functions
├── pipeline.py                     # CLI for running experiments
│
├── test_config.py                  # Fast config testing script (sbatch)
├── test_config.sbatch              # SLURM job for test_config.py
├── run_demo_sim.py                 # Full demographic simulation script
├── run_demo_sim.sbatch             # SLURM job for run_demo_sim.py
│
├── gss_question_lists/             # GSS question filters
│   ├── public_issues_filtered_2021_2024.csv
│   ├── private_life_filtered_2021_2024.csv
│   ├── gss_all_variables.csv
│   ├── gss_demographic_variables.csv    # 104 vars (core/expanded demographic fields)
│   └── gss_politicized_lifestyle_variables.csv  # 67 lifestyle vars (optional)
│
├── data/                           # Raw survey data (ANES)
│   ├── anes_timeseries_2020_csv_20220210.csv
│   ├── anes_timeseries_2024_csv_20250430.csv
│   └── policy_clean.csv
│
├── standalone_test_config/         # Self-contained package for bare metal server
│   ├── test_config.py              # All functions inlined, HuggingFace model loading
│   ├── setup.sh                    # Venv + dependency install
│   └── data/                       # Copies of all CSV/DTA/polarization files
│
├── llm_results/                    # Generated outputs (git-ignored)
└── logs/                           # SLURM job logs
```

## Quick Start

### test_config.py (fast config testing)

Test different configurations quickly with 1% sampling:

```bash
# Default: all fields, prompt B, Llama 3.1, H100
sbatch test_config.sbatch

# With Gemma 2
MODEL_PATH=/project/jevans/maxzhuyt/models/gemma-2-9b-it MODEL_NAME=gemma-2-9b-it sbatch test_config.sbatch

# With lifestyle variables
INCLUDE_LIFESTYLE=1 sbatch test_config.sbatch

# All 2x2x2 combinations (model x fields x lifestyle)
DEMO_FIELDS=core sbatch test_config.sbatch
DEMO_FIELDS=expanded sbatch test_config.sbatch
DEMO_FIELDS=core INCLUDE_LIFESTYLE=1 sbatch test_config.sbatch
DEMO_FIELDS=expanded INCLUDE_LIFESTYLE=1 sbatch test_config.sbatch
# (repeat with MODEL_PATH/MODEL_NAME for Gemma)
```

Environment variable overrides: `MODEL_PATH`, `MODEL_NAME`, `DEMO_FIELDS`, `PROMPT_FMT`, `SAMPLE_FRAC`, `BATCH_SIZE`, `INCLUDE_LIFESTYLE`.

### run_demo_sim.py (full production runs)

```bash
sbatch run_demo_sim.sbatch
```

### Standalone (bare metal RTX 3090)

```bash
cd standalone_test_config
bash setup.sh
source venv/bin/activate
python test_config.py --model-id google/gemma-2-9b-it --model-name gemma-2-9b-it --batch-size 8
```

## Methodology

### Party Coding
- Democrat (100): partyid 0 (Strong D), 1 (Not strong D), 2 (Ind near D)
- Republican (200): partyid 4 (Ind near R), 5 (Not strong R), 6 (Strong R)
- Excluded: partyid 3 (pure independent)

### Survey Polarization
Partisan polarization = normalized distance between Democrat and Republican responses:
1. Min-max normalize responses to [0, 1]
2. Compute mean for each party
3. Polarization = |mean_R - mean_D|

### LLM Polarization
1. Generate prompts with GSS demographic personas (age, race, education, political views, etc.)
2. Extract attention head activations (all layers, all heads)
3. PCA to 15 dimensions, then Mahalanobis distance between D/R centroids
4. Per-topic overlap exclusion: if survey question variable is also a persona field, exclude it

### Correlation
Compare LLM activation polarization with survey polarization across topics (Pearson r, Spearman rho).

## Output Files

`test_config.py` saves incrementally:
- `test_config_detail_{tag}_{model}_{timestamp}_{category}.pkl` - per-category results
- `test_config_detail_{tag}_{model}_{timestamp}_checkpoint.pkl` - running checkpoint
- `test_config_detail_{tag}_{model}_{timestamp}.pkl` - final complete results
- `test_config_{tag}_{model}_{timestamp}.csv` - correlation summary

Load partial results in a notebook: `df = pd.read_pickle('path_to_checkpoint.pkl')`

## GPU Clusters

| Partition | GPUs | VRAM | Recommended batch_size |
|-----------|------|------|----------------------|
| `jevans-gpu` | 4x H100 | 80 GB | 32-96 |
| `ssd-gpu` | 4x A100 | 40 GB | 16-24 |
| Bare metal | 1x RTX 3090 | 24 GB | 4-8 |

## Requirements

```
pandas
numpy
matplotlib
torch
transformers
scipy
scikit-learn
joblib
```
