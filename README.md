# ANES 2020 Policy Polarization Analysis

This project analyzes policy preferences and partisan polarization using data from the American National Election Studies (ANES) 2020 Time Series Study.

## Overview

The analysis examines how different policy issues vary in public opinion and how polarized they are between Democrats and Republicans. By calculating both variance (overall disagreement) and polarization (partisan disagreement), we can identify which issues divide Americans most deeply.

## Data Source

- **Dataset**: ANES 2020 and 2024 Time Series Study
- **Raw Data**: `data/anes_timeseries_2020_csv_20220210.csv`
- **Codebook**: `data/anes_timeseries_2020_userguidecodebook_20220210.txt`

## Policy Column Selection and Transformation

### Selection Criteria

Policy variables were selected based on the following criteria:
1. **Closed-ended questions** with defined response scales 
2. **Substantive policy content** covering major political issues
3. **Non-demographic** variables (ideology and party ID are tracked separately)

### Transformation Methodology

All policy variables were transformed to a consistent directional scale where possible:

**Simple Directional Mapping**: Variables with clear liberal/conservative direction were mapped to signed scales (e.g., -1 for conservative, +1 for liberal)


### Normalization

For cross-policy comparison, all policy columns were min-max normalized to [0, 1]:

```
normalized_value = (value - min) / (max - min)
```

**Note**: Ideology and party ID variables were excluded from normalization to serve as independent grouping variables.

## Analysis Results

### 1. Policy Variance

Variance measures overall disagreement on each issue across all respondents, indicating which policies lack consensus.

**High variance policies** show Americans are divided across the political spectrum on these issues, with opinions distributed widely rather than clustered around a single position.

### 2. Policy Polarization by Party

Polarization is measured using **Mahalanobis distance** between Republican and Democratic response distributions. This accounts for variance and covariance structure, providing a more robust measure than simple mean differences.

**Top Polarized Policy Areas:**
- Border wall / Immigration
- Abortion-related issues
- Biden approval (economy, crime, foreign policy)
- Climate/environment
- Obamacare / Health care
- LGBTQ rights

## Files

- `main.ipynb` - Main analysis notebook
- `data/policy_clean.csv` - Cleaned and transformed policy dataset
- `data/extracted_questions.txt` - Extracted questions from ANES codebook
- `policy_questions.md` - List of policy questions analyzed

## Requirements

```
pandas
numpy
matplotlib
pymupdf
tiktoken
```

## Usage

Run the cells in `main.ipynb` sequentially to:
1. Extract questions from the ANES codebook
2. Clean and transform policy variables
3. Calculate variance and polarization metrics

## Notes

- All analysis excludes respondents with missing data on specific variables
- Party ID is coded as: -1 (Democrat), 0 (Independent), +1 (Republican)
- The analysis focuses on policy preferences, not voting behavior or candidate evaluations

---

# LLM Polarization Analysis

This project also analyzes how Large Language Model (LLM) head-wise activation polarization correlates with the ANES survey data above.

## Quick Start

```bash
# Run both politician and ideology analyses
python pipeline.py --both

# Or run separately
python pipeline.py --politician
python pipeline.py --ideology --n-per-level 50
```

## LLM Analysis File Structure

| File | Description |
|------|-------------|
| `config.py` | Topics, model paths, ideology labels, system messages |
| `model_utils.py` | Model loading and activation extraction |
| `metrics_utils.py` | Polarization metrics calculation |
| `prompt_utils.py` | Prompt generators for both prompt types |
| `pipeline.py` | Main CLI script |
| `analysis_example.ipynb` | Example notebook for experimentation |

## Two Prompt Strategies

### 1. Politician Prompts
Uses real U.S. politician names from NOMINATE data. Each politician has a known party label (Democrat=100, Republican=200).

```
"Generate a statement by Nancy Pelosi on climate change."
"Generate a statement by Mitch McConnell on climate change."
...
```

**Sample size**: ~500 politicians (depending on NOMINATE file)

### 2. Ideology Prompts (7-Point Scale)
Uses generic ideological labels instead of specific politicians.

```
"Imagine you are an American voter who identifies as extremely liberal. What is your view on climate change?"
"Imagine you are an American voter who identifies as liberal. What is your view on climate change?"
...
"Imagine you are an American voter who identifies as extremely conservative. What is your view on climate change?"
```

## `N_PER_LEVEL`

The `N_PER_LEVEL` parameter controls how many **different prompt templates** are used per ideology level per topic.

### Why template variation matters

LLM forward passes are **deterministic** - the same prompt always produces the same activations. Simply repeating identical prompts would give duplicate data points.

Instead, we cycle through 20 different prompt phrasings:
```
"Imagine you are an American voter who identifies as {ideology}. What is your view on {topic}?"
"As someone who is {ideology}, what do you think about {topic}?"
"Speaking as a {ideology} voter, what's your position on {topic}?"
...
```


### Example

With `N_PER_LEVEL=10` for the topic "abortion":
```
# Level 1 (extremely liberal) gets 10 different prompts:
"Imagine you are an American voter who identifies as extremely liberal. What is your view on abortion?"
"As someone who is extremely liberal, what do you think about abortion?"
"Speaking as a extremely liberal voter, what's your position on abortion?"
... (7 more variants)

# Level 2 (liberal) gets the same 10 templates with "liberal":
"Imagine you are an American voter who identifies as liberal. What is your view on abortion?"
...
```

## Experimenting with Prompt Templates

Both prompt types support custom templates:

```python
from pipeline import run_politician_pipeline, run_ideology_pipeline

# Custom politician template
df = run_politician_pipeline(
    model, tokenizer,
    template="Write a tweet from {name} about {topic}."
)

# Custom ideology template
df = run_ideology_pipeline(
    model, tokenizer,
    template="As a {ideology} American, share your thoughts on {topic}."
)
```

### Built-in Templates

**Politician templates** (`prompt_utils.POLITICIAN_TEMPLATES`):
- `default`: "Generate a statement by {name} on {topic}."
- `opinion`: "What would {name} say about {topic}?"
- `tweet`: "Write a tweet from {name} about {topic}."

**Ideology templates** (`prompt_utils.IDEOLOGY_TEMPLATES`):
- `default`: "Imagine you are an American voter who identifies as {ideology}. What is your view on {topic}?"
- `opinion`: "As someone who is {ideology}, what do you think about {topic}?"
- `agree`: "You are a {ideology} American. Do you agree or disagree with policies related to {topic}? Explain."

## Polarization Metrics

We compute multiple metrics to measure separation between liberal and conservative activations in each attention head. **Mahalanobis distance is the primary metric** used for correlation with ANES survey polarization.

### Primary Metric

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Mahalanobis** | Mahalanobis distance between group centroids | Higher = greater separation between liberal/conservative activations, accounting for covariance structure |

### Secondary Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Davies-Bouldin** | Davies-Bouldin clustering index | Lower = better cluster separation (we use reciprocal for consistency) |
| **Total_Dispersion** | Sum of eigenvalues (total variance) | Higher = more variance in activation space |
| **PC1_Ratio** | Explained variance ratio of PC1 | Higher = activations lie along a single dimension |
| **Intrinsic_Dim** | Intrinsic dimensionality (participation ratio) | Lower = activations concentrated in fewer dimensions |
| **Polarization_CoG** | Center of gravity across layers | Higher = polarization signal in later layers |

### Why Mahalanobis?

Mahalanobis distance is preferred because it:
1. **Accounts for covariance** - Unlike Euclidean distance, it considers how features co-vary
2. **Scale-invariant** - Not affected by different scales across activation dimensions
3. **Matches ANES methodology** - ANES polarization is also computed using Mahalanobis distance between party group centroids

## LLM Output Format

Both pipelines output a DataFrame with one row per topic:

| Column | Description |
|--------|-------------|
| `Topic` | Topic key (e.g., "abortion", "gun_bkg_chk") |
| `Avg_Mahalanobis` | Average Mahalanobis distance across all heads **(primary metric)** |
| `Max_Mahalanobis` | Maximum Mahalanobis distance |
| `Avg_Total_Dispersion` | Average total variance |
| `Avg_PC1_Ratio` | Average explained variance of PC1 |
| `Avg_Davies_Bouldin` | Average Davies-Bouldin index |
| `Polarization_CoG` | Center of gravity (which layers show most polarization) |
| `Grid_*` | Full [L, H] grids for each metric |

## Representative Question Selection

For cross-area correlation analysis, we select **one representative question per policy area** (18 areas total). The selection logic prioritizes the **broadest issue** in terms of sub-issue coverage within each area.

| Area | Selected Topic | Alternatives | Selection Rationale |
|------|---------------|--------------|---------------------|
| **Abortion** | `abortion` | `scotus_abort`, `biden_abortion` | General abortion attitudes cover legality and access broadly |
| **Climate** | `clim_imp` | `env_bus`, `ghg_emiss` | Importance of climate change as an issue encompasses regulation and environmental tradeoffs |
| **CrimeAndPolicing** | `biden_crime` | `death_pen`, `police_force` | Biden's handling of crime is a broad umbrella for policing and criminal justice policy |
| **Economy** | `biden_economy` | — | Only economy question available |
| **Diversity** | `diversity` | `assist_black`, `black_favor` | General diversity attitudes cover racial attitudes broadly |
| **Education** | `dei_college` | `spend_school`, `affirm_action` | DEI policies encompass diversity, inclusion, and higher education issues |
| **Elections** | `vote_denied` | `voter_id`, `felon_vote` | Perceived voting denial captures broader election integrity concerns |
| **ForeignPolicy** | `biden_foreign` | `def_spend`, `mil_force` | Biden's foreign policy handling includes war, military spending, and foreign intervention |
| **Gender** | `ft_fem` | — | Only gender-specific question available (feeling thermometer toward feminists) |
| **Guns** | `gun_bkg_chk` | `ar_ban`, `gun_imp` | Background checks are a widely debated and representative gun policy measure |
| **Health** | `govt_health` | `vax_school`, `obamacare` | Government-provided health insurance is a fundamental health policy question |
| **Immigration** | `biden_immigration` | `imm_unauth`, `border_wall` | Biden's immigration handling covers unauthorized immigration, border policy, etc. |
| **Institutions** | `checks_power` | `trump_corr`, `journ_access` | Checks and balances are broader than specific corruption or press access concerns |
| **Labor** | `min_wage` | `paid_leave`, `job_gov_guar` | ⚠️ *Uncertain choice*: minimum wage does not cover job guarantee or parental leave |
| **Lesbian/Gay** | `lg_marry` | `lg_job`, `lg_refuse_service` | Same-sex marriage is a landmark and broadly representative LGBTQ issue |
| **Redistribution** | `spend_poor` | `spend_welfare`, `millionaire_tax` | Spending to aid the poor is a direct redistribution measure |
| **Trade** | `free_trade` | `intl_trade_job`, `limit_imports` | Free trade support is the broadest trade policy question |
| **Transgender** | `ft_trans` | `trans_bath`, `trans_military` | Feeling thermometer toward transgender people is the broadest attitude measure |

### Notes on Uncertain Selections

- **Labor (`min_wage`)**: The minimum wage question is narrow compared to alternatives like job guarantee (`job_gov_guar`) or paid parental leave (`paid_leave`). None of the labor questions fully covers the breadth of labor policy issues.
- **Institutions (`checks_power`)**: No single question captures all institutional concerns (corruption, press freedom, scientific trust). Checks of power was selected as the most fundamental institutional principle.

## Correlation Results

### LLM vs ANES Polarization (Mahalanobis Distance)

| Dataset | Prompt Type | Pearson r | Spearman ρ |
|---------|-------------|-----------|------------|
| **All 65 Issues** | Politician | 0.304 | 0.344 |
| **All 65 Issues** | Ideology | 0.115 | 0.141 |
| **18 Representative Issues** | Politician | **0.597** | **0.653** |
| **18 Representative Issues** | Ideology | 0.090 | -0.038 |

### Key Findings

1. **Politician prompts outperform ideology prompts** - Using real politician names produces LLM activations that better correlate with human survey polarization.

2. **Representative issue selection improves correlation** - Selecting one broad issue per policy area (18 topics) nearly doubles the Pearson correlation (0.304 → 0.597) compared to using all 65 issues.

3. **Ideology prompts show weak/no correlation** - Generic ideological labels ("extremely liberal", "conservative", etc.) do not produce activation patterns that correlate with ANES polarization.

## Comparing LLM Results with ANES

```python
import pandas as pd

# Load results
df_pol = pd.read_pickle("df_llm_politician_TIMESTAMP.pkl")
df_ideo = pd.read_pickle("df_llm_ideology_TIMESTAMP.pkl")
df_anes = pd.read_csv("policy_polarization.csv")

# Merge and correlate
df_merged = df_pol[["Topic", "Avg_Mahalanobis"]].merge(
    df_anes.rename(columns={"issue": "Topic", "mahalanobis_distance": "ANES_Mahalanobis"}),
    on="Topic"
)

correlation = df_merged["Avg_Mahalanobis"].corr(df_merged["ANES_Mahalanobis"])
print(f"Correlation: {correlation:.3f}")
```
