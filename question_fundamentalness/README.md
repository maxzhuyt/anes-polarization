# Question Fundamentalness Analysis

Identifies which GSS survey questions are most "fundamental" to understanding political polarization using five independent methods.

## Methods

1. **Mutual Information** - Which questions reduce uncertainty about others?
2. **Predictive Power** - Which questions predict responses to others (out-of-sample)?
3. **Network Centrality** - Which questions bridge different opinion clusters?
4. **Dimensionality Reduction** - Which questions capture major axes of variation (PCA/FA)?
5. **Tree Structure** - Which questions sit near the root of the optimal dependency tree (Chow-Liu)?

## Running the Analysis

```bash
cd /project/jevans/maxzhuyt/gss_polarization/question_fundamentalness

# Run individual methods
python scripts/run_mutual_information.py --n_jobs 14 --output_prefix results/mi
python scripts/run_network_centrality.py --n_jobs 14 --output_prefix results/network
python scripts/run_dimensionality.py --output_prefix results/dimensionality
python scripts/run_tree_structure.py --n_jobs 14 --output_prefix results/tree
python scripts/run_predictive_power.py --n_jobs 14 --output_prefix results/predictive

# Or run all at once
python scripts/run_all_analyses.py --n_jobs 14 --output_dir results
```

## Notebooks (in `notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `gss_preprocessing.ipynb` | Create combined GSS dataset from Stata files |
| `gss_new.ipynb` | Compute polarization scores by topic |
| `cog_fundamentalness_correlation.ipynb` | Correlate fundamentalness with LLM polarization |
| `cog_fundamentalness_correlation_v2.ipynb` | Extended correlation analysis |

## Output Files

```
results/
├── combined_hierarchy.csv       # Unified ranking with ensemble scores
├── mi_scores.csv, mi_mi_matrix.csv, mi_nmi_matrix.csv
├── predictive_scores.csv, predictive_acc_matrix.csv, predictive_r2_matrix.csv
├── network_scores.csv, network_correlation_matrix.csv, network_geodesic_matrix.csv
├── dimensionality_scores.csv, dimensionality_pca_loadings.csv
├── tree_scores.csv, tree_edges.csv
└── questions.csv
```

## Data

- **Source**: GSS 2021, 2022, 2024 (11,066 respondents, 138 public opinion questions)
- **Filtering**: Questions appearing in at least 2 of 3 survey years (98 questions)
- Data loaded from `../data/gss/`
