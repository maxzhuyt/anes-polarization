# LLM Polarization Analysis

Measuring whether the partisan geometry in LLM attention head activations tracks real political polarization as measured by GSS survey responses.

The core question: when topic A generates more partisan separation in LLM activations than topic B, is that the same ordering as in actual survey data?

---

## Directory Structure

```
llm_polarization/
├── pipelines/                    # Production pipelines
│   ├── politician_simulation/    # Politician-based polarization analysis
│   │   ├── run_gss_pca.py        # Original pipeline
│   │   ├── run_gss_pca_v2.py     # Updated with prompt framing & layer selection
│   │   └── *.sbatch
│   ├── demographic_simulation/   # Demographic-based polarization analysis
│   │   ├── run_demo_sim.py       # Original pipeline
│   │   ├── run_demo_sim_v2.py    # Updated with prompt framing & layer selection
│   │   └── *.sbatch
│   └── model_comparison/         # Base vs Instruct vs Reasoning comparison
│       ├── run_model_comparison.py
│       └── *.sbatch
├── experiments/                  # Small-scale exploratory experiments
│   ├── prompt_framing/           # Exp 0A: prompt type effects
│   ├── layer_selection/          # Exp 0B: layer selection effects
│   ├── encoding_strength/        # Exp 1: encoding strength analysis
│   ├── large_models/             # Large model experiments
│   ├── dolphin_models/           # Dolphin fine-tuned model experiments
│   └── ...                       # Other experiments
├── notebooks/                    # Jupyter notebooks for analysis
├── colab_demo_dolphin/           # Colab notebooks
├── results/                      # Main pipeline results
├── logs/                         # SLURM job logs
└── [utils]                       # config.py, model_utils.py, etc.
```

---

## Two Simulation Methods

### 1. Politician Simulation (`pipelines/politician_simulation/`)

**Who**: 550 members of the 116th U.S. Congress, loaded from `data/politicians.csv`.
- Democrats: party_code=100 (`partyid` codes 0–2 in the NOMINATE data)
- Republicans: party_code=200 (`partyid` codes 200 in the NOMINATE data)
- Names via `fullname` column; excluded any row missing `bioname`

**What**: For each (politician, topic) pair, a single prompt is constructed and the model's activations at the last token are extracted. There are three prompt conditions:

| Condition | Instruct template | Base template |
|-----------|------------------|---------------|
| `rhetorical` | `Generate a statement by {name} on {topic}.` | `{name} makes a statement on {topic}:` |
| `stance` | `What is {name}'s position on {topic}?` | `On {topic}, {name}'s position is` |
| `survey` | `If asked in a national survey about {question}, how would {name} respond?` | `In a survey about {question}, {name} would respond` |

`{topic}` = **NaturalLanguageClause** from `question_lists/public_issues.csv` (e.g., `"whether a pregnant woman should be able to obtain a legal abortion for any reason"`).
`{question}` = same NaturalLanguageClause (with SurveyQuestion as fallback).

System message: `"You are simulating the public stance of U.S. politicians."`

Max sequence length: 128 tokens.

**What "outperforming demo" means**: The politician simulation achieves a higher Pearson r between LLM activation polarization and GSS survey polarization than the demographic simulation. This means the model's implicit political knowledge (encoded when prompted with actual politician names) more accurately mirrors which issues real Americans are polarized on, compared to the model's behavior when given explicit demographic personas.

---

### 2. Demographic Simulation (`pipelines/demographic_simulation/`)

**Who**: GSS 2021–2024 survey respondents (real people), filtered to Democrats and Republicans:
- Democrat (party_code=100): `partyid` ∈ {0=Strong D, 1=Not strong D, 2=Ind near D}
- Republican (party_code=200): `partyid` ∈ {4=Ind near R, 5=Not strong R, 6=Strong R}
- Excluded: `partyid=3` (pure independent)

For each topic, only respondents who gave a valid response to that topic are included, with a minimum of 10 Democrats and 10 Republicans. In `test_config.py` / `run_demo_sim.py`, a 10% stratified sample is drawn per topic (stratified on `polviews, age_bin, degree, race, sex, rincome`); `exp_large_models.py` uses a fixed 10% stratified sample drawn once (`DEMO_SAMPLE_FRAC = 0.1`).

**Profile construction**: For each respondent, 83 demographic variables (from `gss_demographic_variables.csv`) are read. Each field's numeric code is mapped to a human-readable label via Stata value labels (loaded from `GSS2024.dta` and `GSS2022.dta`). Fields with missing values are skipped. Fields are joined with ". " and **shuffled randomly** (seeded per-topic on `hash(topic_name) % 2^32` to prevent ordering artifacts).

Example profile: `"Age: 45. Education: Bachelor's degree. Race: White. Sex: Female. Marital status: Married. [...]"`

**Prompt format (default B, used in production)**:
```
System: "You are simulating the views of an American."
User:   "Given the following background about a person:
         {profile}

         How would they answer: {survey_question}"
```
Where `{survey_question}` = **SurveyQuestion** column from `question_lists/public_issues.csv` (the verbatim or lightly cleaned GSS question wording).

The three format variants (relevant to `test_config.py`):
- **Format A**: `{profile}\n\nSurvey question: {question}`
- **Format B** (default): `Given the following background about a person:\n{profile}\n\nHow would they answer: {question}`
- **Format C**: `{profile}\n\n{question}`

Max sequence length: 512 tokens (longer because profiles are ~400–600 chars).

**Leakage prevention**: If `--include-lifestyle` is used and a lifestyle variable coincides with the survey topic being asked, that field is excluded from the persona for that specific topic.

---

## Activation Extraction

Hooks are registered on `model.model.layers[l].self_attn.o_proj` for all L layers. The hook captures `input[0]` — the pre-projection concatenated head outputs — at each forward pass.

- Raw hook shape: `[Batch, Seq, Hidden]`
- Reshaped to: `[Batch, Seq, H, D_head]` where `H = num_attention_heads`, `D_head = hidden_size / H` (or explicit `head_dim` for GQA models like Gemma)
- **Last token only**: `[:, -1, :, :]` → shape `[Batch, H, D_head]`
  - Why last token: with left-padding, the final position has attended to the full prompt, making it the richest summary representation.

After all batches: concatenated to `[N, L, H, D]`.

---

## Per-head Mahalanobis (PCA-based)

For each of the L×H attention heads independently:

1. Extract `X = activations[:, l, h, :]` — shape `(N, D)`, typically D=128.
2. **PCA**: reduce to `n_components = min(15, D, N−1)` dimensions using `sklearn.PCA`. This is fit on all N samples (both parties pooled), then transforms all N.
3. **Split by party**: D-set = rows where label=100; R-set = rows where label=200.
4. **Centroids**: weighted mean (default) or weighted median of the PCA-reduced points.
5. **Pooled covariance**: `cov_pool = (cov_dem + cov_rep) / 2 + 1e-6 × I` (regularized, to handle near-singular cases).
6. **Mahalanobis**: `sqrt((μ_D − μ_R)^T · cov_pool^{−1} · (μ_D − μ_R))`.
   - Requires at least 6 samples per party; returns 0.0 if fewer or if inversion fails.

This yields a grid of shape `(L, H)`.

**Summary metrics** (per topic, per condition):
- `mahal_all` = `mean(grid)` — average over all L×H heads
- `mahal_mid10` = `mean(grid[L×0.45 : max(L×0.55, L×0.45+1), :])` — average over the middle 10% of layers only
  - For a 32-layer model: layers 14–17; for an 80-layer model: layers 36–43
  - Finding: `mahal_mid10` hurts large/deep models (70–80 layers), deltas as bad as −0.48 in r

`mahal_max` (demographic only) = `max(grid)` — the single most discriminative head.

---

## GSS Polarization Measure

For each topic, the survey-measured polarization is:

```
polarization = |mean_rep − mean_dem| / scale_range
```

where `mean_dem` and `mean_rep` are the average response codes among Democrat and Republican respondents respectively, and `scale_range` is the maximum response code on that question's scale (e.g., 1.0 for binary questions, 6.0 for 7-point scales). This normalizes all topics to [0, 1] regardless of original scale. Minimum thresholds: ≥100 Democrats, ≥100 Republicans, ≥200 total respondents in GSS.

Stored in `data/polarization/public_issues_polarization.csv` and `data/polarization/private_life_polarization.csv`.

---

## Final Alignment Score

For each simulation method × layer selection × centroid method, compute across all N topics:

- **Pearson r**: linear correlation between LLM `mahal_all` and GSS `polarization`
- **Spearman ρ**: rank correlation (more robust to outliers)

N = 126 filtered public issues (full list minus 8 topics containing racial comparison framing: `hubbywk1`, `racdif1–4`, `workwhts`, `wlthwhts`, `intlwhts`). The correlation is across topics, not respondents.

---

## Party Coding

| partyid | Label | party_code |
|---------|-------|------------|
| 0 | Strong Democrat | 100 |
| 1 | Not strong Democrat | 100 |
| 2 | Independent near Democrat | 100 |
| 3 | Independent | excluded |
| 4 | Independent near Republican | 200 |
| 5 | Not strong Republican | 200 |
| 6 | Strong Republican | 200 |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `test_config.py` | Fast config testing (1% sample, various field/prompt combos) |
| `run_demo_sim.py` | Full demographic simulation (all respondents) |
| `run_gss_pca.py` | Politician simulation + PCA/Mahalanobis + bootstrap |
| `run_model_comparison.py` | Base vs instruct vs reasoning model comparison |
| `pipeline.py` | CLI for running experiments |
| `model_utils.py` | Model loading + `extract_heads_batched` (activation extraction) |
| `prompt_utils.py` | Politician and ideology prompt generation |

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
| Exp Large | `exp_large_models.py` | Politician + demo sim for 7 large models (24B–72B) | In progress |

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

# Large model (one job per model)
sbatch experiments/exp_large_mistral24b.sbatch
```

Environment variable overrides: `MODEL_PATH`, `MODEL_NAME`, `DEMO_FIELDS`, `PROMPT_FMT`, `SAMPLE_FRAC`, `BATCH_SIZE`, `INCLUDE_LIFESTYLE`.

## Output Files

`test_config.py` saves incrementally:
- `test_config_detail_{tag}_{model}_{timestamp}_{category}.pkl` - per-category results
- `test_config_detail_{tag}_{model}_{timestamp}_checkpoint.pkl` - running checkpoint
- `test_config_detail_{tag}_{model}_{timestamp}.pkl` - final complete results
- `test_config_{tag}_{model}_{timestamp}.csv` - correlation summary

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
- Politicians: `/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv`
- Models: `/project/jevans/maxzhuyt/models/`
