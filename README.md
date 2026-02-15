# GSS Polarization Project

Analyzing political polarization in the United States using the General Social Survey (GSS) and LLM activation analysis.

## Project Structure

```
gss_polarization/
├── data/                              # All raw data
│   ├── gss/                           # GSS survey data (2021-2024)
│   │   ├── gss_2021_2024.csv          # Combined GSS dataset
│   │   ├── GSS2022.dta                # Stata format
│   │   └── GSS2024.dta
│   ├── anes/                          # ANES survey data
│   │   ├── anes_timeseries_2020_*.csv
│   │   ├── anes_timeseries_2024_*.csv
│   │   └── policy_clean.csv
│   ├── politicians.csv                # DW-NOMINATE data (116th Congress, Voteview)
│   └── polarization/                  # Survey polarization scores
│       ├── public_issues_polarization.csv
│       ├── private_life_polarization.csv
│       └── (filtered variants)
│
├── question_lists/                    # GSS variable metadata
│   ├── gss_demographic_variables.csv  # 104 demographic variables
│   ├── gss_politicized_lifestyle_variables.csv  # 67 lifestyle variables
│   ├── gss_all_variables.csv
│   ├── public_issues.csv
│   └── private_life.csv
│
├── llm_polarization/                  # LLM activation polarization analysis
│   ├── README.md
│   ├── config.py, model_utils.py, metrics_utils.py, prompt_utils.py
│   ├── test_config.py, run_demo_sim.py, run_gss_pca.py, pipeline.py
│   ├── *.sbatch                       # SLURM job scripts
│   ├── notebooks/                     # Jupyter notebooks
│   ├── results/                       # LLM output files
│   └── logs/
│
├── question_fundamentalness/          # Question hierarchy analysis
│   ├── README.md
│   ├── gss_topic_mappings.py
│   ├── methods/                       # Analysis methods (MI, predictive, network, PCA, tree)
│   ├── scripts/                       # Standalone analysis scripts
│   ├── notebooks/                     # Jupyter notebooks
│   ├── results/                       # Analysis outputs
│   └── logs/
│
├── standalone_test_config/            # Self-contained package for bare metal server
│   ├── test_config.py
│   ├── setup.sh
│   └── data/
│
└── archive/                           # Legacy analyses from both projects
```

## Modules

### LLM Polarization (`llm_polarization/`)
Tests whether LLM attention head activations reflect survey-measured partisan polarization. Uses demographic persona prompts → activation extraction → PCA + Mahalanobis distance between Democrat/Republican centroids.

See [llm_polarization/README.md](llm_polarization/README.md) for details.

### Question Fundamentalness (`question_fundamentalness/`)
Identifies which survey questions are "fundamental" using five independent methods: mutual information, predictive power, network centrality, dimensionality reduction, and tree structure analysis.

See [question_fundamentalness/README.md](question_fundamentalness/README.md) for details.

## Data Sources

- **GSS**: General Social Survey 2021, 2022, 2024 (Smith et al., NORC at the University of Chicago)
- **ANES**: American National Election Studies 2020, 2024

## GPU Clusters

| Partition | GPUs | VRAM | Recommended batch_size |
|-----------|------|------|----------------------|
| `jevans-gpu` | 4x H100 | 80 GB | 32-96 |
| `ssd-gpu` | 4x A100 | 40 GB | 16-24 |
| Bare metal | 1x RTX 3090 | 24 GB | 4-8 |

## Requirements

```
pandas, numpy, matplotlib, torch, transformers, scipy, scikit-learn, joblib, networkx, seaborn
```
