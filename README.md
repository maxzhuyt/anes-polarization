# LLM Polarization Analysis

Analyzing how LLM activation polarization correlates with survey-measured partisan polarization (GSS and ANES).

## Project Structure

```
├── analysis_gss.ipynb           # Main analysis: GSS single-model
├── analysis_gss_multimodel.ipynb # Main analysis: GSS multi-model comparison
├── analysis_anes.ipynb          # Supplementary: ANES analysis
│
├── config.py                    # Topics, model paths, settings
├── model_utils.py               # Model loading, activation extraction
├── metrics_utils.py             # Polarization metrics (Mahalanobis, etc.)
├── prompt_utils.py              # Prompt generators
├── pipeline.py                  # CLI for running experiments
│
├── gss_question_lists/          # GSS question filters
│   ├── public_issues_filtered_2021_2024.csv
│   ├── private_life_filtered_2021_2024.csv
│   ├── gss_all_variables.csv
│   └── gss_demographic_variables.csv
│
├── data/                        # Raw survey data (ANES)
│   ├── anes_timeseries_2020_csv_20220210.csv
│   ├── anes_timeseries_2024_csv_20250430.csv
│   └── policy_clean.csv
│
└── llm_results/                 # Generated outputs (git-ignored)
```

## Quick Start

```bash
# Run politician-based analysis
python pipeline.py --politician

# Run ideology-based analysis
python pipeline.py --ideology --n-per-level 50

# Run both
python pipeline.py --both
```

## Methodology

### Survey Polarization
Partisan polarization = normalized distance between Democrat and Republican responses:
1. Min-max normalize responses to [0, 1]
2. Compute mean for each party
3. Polarization = |μ_R − μ_D|

### LLM Polarization
1. Generate prompts with politician names or ideology labels
2. Extract attention head activations
3. Compute Mahalanobis distance between liberal/conservative activation centroids

### Correlation
Compare LLM activation polarization with survey polarization across topics.

## Key Results

| Survey | Pearson r |
|--------|-----------|
| GSS (120 questions) | 0.598 |
| ANES (18 broad questions) | 0.652 |

## Requirements

```
pandas
numpy
matplotlib
torch
transformers
```
