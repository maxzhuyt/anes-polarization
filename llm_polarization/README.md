# LLM Polarization Analysis

Analyzing how LLM activation polarization correlates with survey-measured partisan polarization (GSS and ANES).

## How It Works

1. Generate prompts with GSS demographic personas (age, race, education, political views, etc.)
2. Extract attention head activations (all layers, all heads)
3. PCA to 15 dimensions, then Mahalanobis distance between Democrat/Republican centroids
4. Compare LLM activation polarization with survey polarization across topics (Pearson r, Spearman rho)

## Party Coding
- Democrat (100): partyid 0 (Strong D), 1 (Not strong D), 2 (Ind near D)
- Republican (200): partyid 4 (Ind near R), 5 (Not strong R), 6 (Strong R)
- Excluded: partyid 3 (pure independent)

## Scripts

| Script | Purpose |
|--------|---------|
| `test_config.py` | Fast config testing (1% sample, various field/prompt combos) |
| `run_demo_sim.py` | Full demographic simulation |
| `run_gss_pca.py` | PCA + Mahalanobis analysis with politician prompts |
| `run_model_comparison.py` | Base vs instruct vs reasoning model comparison |
| `pipeline.py` | CLI for running experiments |

## Quick Start

```bash
# Default: all fields, prompt B, Llama 3.1, H100
sbatch test_config.sbatch

# With Gemma 2
MODEL_PATH=/project/jevans/maxzhuyt/models/gemma-2-9b-it MODEL_NAME=gemma-2-9b-it sbatch test_config.sbatch

# With lifestyle variables
INCLUDE_LIFESTYLE=1 sbatch test_config.sbatch

# Full production run
sbatch run_demo_sim.sbatch

# Model comparison (base vs instruct vs reasoning)
sbatch run_model_comparison.sbatch
FAMILIES="Qwen3-4B" sbatch run_model_comparison.sbatch
```

Environment variable overrides: `MODEL_PATH`, `MODEL_NAME`, `DEMO_FIELDS`, `PROMPT_FMT`, `SAMPLE_FRAC`, `BATCH_SIZE`, `INCLUDE_LIFESTYLE`.

## Experiments (in `experiments/`)

Modular experiment framework with shared utilities (`shared_utils.py`). Each experiment has a `.py` script and `.sbatch` file.

| Experiment | Script | Description | Status |
|-----------|--------|-------------|--------|
| Exp 1 | `exp1_encoding_strength.py` | Prompt-end encoding strength (base vs instruct vs reasoning) | Done |
| Exp 1R | `exp1_encoding_strength.py` (REPLICATION=True) | Replication with disjoint topics | Done |
| Exp 2 | `exp2_layer_depth.py` | Layer-depth analysis of partisan signal | Done |
| Exp 3 | `exp3_coherence.py` | Cross-topic representational coherence | Done |
| Exp 5 | `exp5_elite_amplification.py` | Elite amplification vs GSS polarization | Done |
| Exp 6 | `exp6_false_polarization.py` | False polarization detection | Done |
| Exp 7 | `exp7_head_discriminability.py` | Per-head LDA discriminability | Done |
| Exp 8 | `exp8_affective_polarization.py` | Affective vs policy polarization | Done |
| Bramson | `exp_bramson_dimensions.py` | Bramson 9 dimensions of polarization | Done |
| Exp 9 | `exp9_anonymization.py` | Name anonymization test | Done |
| Exp 10 | `exp10_residual_signal.py` | Residual topic-specific signal | Done |
| Exp 12 | `exp12_nominate_probing.py` | DW-NOMINATE per-head ridge regression (Kaplan et al. replication) | Done |
| Exp 13 | `exp13_behavioral_validation.py` | Behavioral validation via generation + keyword scoring | Done* (7/8 models) |

Results and analysis: `experiments/RESULTS_ANALYSIS.md`

## Notebooks (in `notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `analysis_gss.ipynb` | GSS single-model analysis |
| `analysis_gss_multimodel.ipynb` | Multi-model comparison |
| `analysis_gss_pca.ipynb` | PCA-based analysis |
| `analysis_anes.ipynb` | ANES analysis |
| `demographic_simulation.ipynb` | Demographic persona simulation |
| `explore_demo_sim.ipynb` | Explore simulation results |
| `explore_test_config.ipynb` | Explore test_config results, topic exclusion analysis |
| `explore_model_comparison.ipynb` | Analyze base vs instruct vs reasoning comparison results |
| `test_config.ipynb` | Interactive config testing (superseded by .py) |

## Output Files

`test_config.py` saves incrementally:
- `test_config_detail_{tag}_{model}_{timestamp}_{category}.pkl` - per-category results
- `test_config_detail_{tag}_{model}_{timestamp}_checkpoint.pkl` - running checkpoint
- `test_config_detail_{tag}_{model}_{timestamp}.pkl` - final complete results
- `test_config_{tag}_{model}_{timestamp}.csv` - correlation summary

Load partial results: `df = pd.read_pickle('results/path_to_checkpoint.pkl')`

`run_model_comparison.py` saves incrementally:
- `comparison_{model_name}_{timestamp}.pkl` - per-model checkpoint
- `comparison_all_{timestamp}_checkpoint.pkl` - running aggregate checkpoint
- `comparison_detail_{timestamp}.csv` - per-topic results (all models)
- `comparison_correlations_{timestamp}.csv` - correlation summary by model/category
- `comparison_filtered_{timestamp}.csv` - correlations with excluded topics removed
- `comparison_bar_{timestamp}.png` - grouped bar chart (base/instruct/reasoning)
- `comparison_scatter_{timestamp}.png` - scatter grid (model × category)

## Data Dependencies

All data paths are relative to the project root (`../`):
- GSS data: `../data/gss/gss_2021_2024.csv`, `../data/gss/GSS2022.dta`, `../data/gss/GSS2024.dta`
- Question lists: `../question_lists/*.csv`
- Polarization: `../data/polarization/*.csv`
- Models: `/project/jevans/maxzhuyt/models/`
