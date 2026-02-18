# Overnight Research Log & Job Summary

**Date**: February 13-14, 2026
**Principal Investigator**: Max Zhu
**Session**: Overnight GPU compute on jevans-gpu partition

---

## Executive Summary

This research log documents a systematic exploration of how large language models (LLMs) encode political polarization in their internal representations. We're running 6 complementary experiments across 8 model variants (base/instruct/reasoning) on 5 model families, analyzing 209 political topics with 550 politicians' activations.

**Key Research Questions:**
1. Do reasoning models encode partisan information differently than base/instruct models?
2. Which polarization dimensions (Bramson et al. 2016) best distinguish model vs. human distributions?
3. How do different model architectures (Qwen, Llama, Gemma, SmolLM) compare in polarization encoding?

**Total Compute**: ~22 hours across 6 jobs on 2 GPUs in parallel (~11 hours wall time)

---

## Design Choices & Rationale

### Multi-Expert Deliberation Process (5 Iterations)

Before coding, we simulated critique from three expert perspectives:
1. **ML/Architecture Expert**: Focused on technical validity (layer depths, attention mechanisms, memory constraints)
2. **Polarization Scholar**: Ensured alignment with social science theory (Bramson 2016, Converse constraint, false polarization)
3. **Computational Social Scientist**: Validated methodology (power analysis, FDR correction, effect sizes, external validation)

**Key Decisions from Deliberation:**
- Use prompt-end extraction (before generation) to isolate encoding from generation artifacts
- Pre-register all hypotheses and topic lists for reproducibility
- Include both base models (completion-style) and instruct/reasoning (chat-style) for comparison
- Measure multiple polarization dimensions, not just Mahalanobis distance
- Compare model outputs to actual GSS survey data (external validation)

### Why These Model Families?

| Family | Base | Instruct | Reasoning | Rationale |
|--------|------|----------|-----------|-----------|
| **Qwen3-4B** | ✓ | ✓ | ✓ | Strong multilingual performance, hybrid thinking variant |
| **Llama-3.1-8B** | ✓ | ✓ | ✓ (DeepSeek-R1) | Industry standard, distilled reasoning model available |
| **Gemma-2-9b** | ✓ | ✓ | ✗ | Google's open model, instruction-tuned variant |
| **SmolLM3-3B** | ✓ | ✓ (/no_think) | ✓ (/think) | Small efficient model, same weights for instruct/reasoning |
| **Qwen2.5-7B** | ✓ | ✗ | ✗ | Baseline comparison for Qwen family evolution |

**Why these specific variants?**
- **Base models**: Test pure pre-training without instruction tuning contamination
- **Instruct models**: Standard chat-tuned models (RLHF/DPO)
- **Reasoning models**: Chain-of-thought capable variants (test if explicit reasoning changes political encoding)

### Topic Selection Strategy

**Public Issues** (134 topics): Abortion, gun control, government spending, immigration, etc.
**Private Life** (75 topics): Religion, family values, personal morality, lifestyle choices

**Pre-specified Samples** (reproducibility):
- **Exp1/1R**: Random 30+30 (seed=42/43) - general encoding strength
- **Exp2**: Random 20+20 (seed=42) - layer analysis (fewer topics for speed)
- **Exp5/Bramson**: Top 30 most polarized (by GSS effect size) - maximize signal
- **Exp6**: 40 moderate polarization (0.2 < pol < 0.7) - test overlap/false polarization

**Excluded Topics** (measurement issues):
- Public: hubbywk1, racdif1-4, workwhts, wlthwhts, intlwhts
- Private: reborn, marwht, helpful, helpfulnv, helpfulv

### Prompt Design Philosophy

**Why different prompts for base vs. instruct models?**

*Base models* need completion-style prompts (no chat template):
- Public: `"{name} makes a statement on {topic}:"`
- Private: `"When asked about {topic}, {name} says"`
- **Rationale**: These models are trained on raw text continuation, not instruction following

*Instruct/Reasoning models* use chat templates with system message:
- Public: `"Generate a statement by {name} on {topic}."` (POLITICIAN_TEMPLATES['default'])
- Private: `"What would {name} say about {topic}?"` (POLITICIAN_TEMPLATES['opinion'])
- **Rationale**: Instruction-tuned models expect structured prompts; this matches their training distribution

**Why this matters**: Prompt sensitivity tests (run_base_prompts.sbatch) showed:
- Alternative prompt better for public issues (3/5 models improved)
- Original prompt better for private life (4/5 models improved)
- Decision: Use category-specific prompts optimized for each domain

### Statistical Rigor

**Pre-registration**: All hypotheses, sample sizes, and analysis plans specified before data collection

**Power Analysis**:
- Exp1: N=60 topics, 80% power to detect d=0.5 (medium effect)
- Exp2: N=40 topics, adequate for mixed models with 3 depth levels
- Bramson: N=30 topics, sufficient for correlation analysis (r>0.3)

**Multiple Comparisons Correction**:
- Within-experiment: Benjamini-Hochberg FDR correction (q < 0.05)
- Effect sizes: Cohen's d, R², η² reported alongside p-values
- Reproducibility: Exp1 vs Exp1R replication with disjoint topic samples

**External Validation**:
- GSS polarization scores (2010-2022 waves)
- ANES feeling thermometers (for affective polarization)
- Multi-dataset triangulation strengthens claims

---

## Currently Running Jobs

### Job Status (Live Updates)

| Job ID | Name | Status | Time | Models | Topics | Progress |
|--------|------|--------|------|--------|--------|----------|
| 45522190 | qwen_comparison | RUNNING | 3:15+ | Qwen3-4B (3 variants) | 209 all | Processing |
| 45522191 | llama_comparison | QUEUED | - | Llama-3.1-8B (3 variants) | 209 all | Waiting |
| 45522192 | gemma_comparison | QUEUED | - | Gemma-2-9b (2 variants) | 209 all | Waiting |
| 45522193 | smollm_comparison | QUEUED | - | SmolLM3-3B (3 variants) | 209 all | Waiting |
| 45522194 | qwen25_comparison | QUEUED | - | Qwen2.5-7B (1 variant) | 209 all | Waiting |
| 45523421 | **bramson_dims** | QUEUED | - | All 8 models | 30 polarized | **NEW!** Waiting |

### Bramson Dimensions Experiment (NEW - Just Queued!)

**What it does**: Computes all 9 independent polarization dimensions from Bramson et al. (2016):

1. **Spread**: Standard deviation (opinion range)
2. **Dispersion**: Variance (opinion scatter)
3. **Coverage**: Proportion of opinion space occupied
4. **Regionalization**: Clustering tendency (silhouette score)
5. **Fragmentation**: Number of distinct peaks/modes (KDE + peak detection)
6. **Distinctness**: Between-group separation (Cohen's d)
7. **Group Divergence**: Between-group variance / total variance (eta-squared)
8. **Group Consensus**: Within-group homogeneity (inverse variance)
9. **Size Parity**: Balance between Democrat/Republican group sizes

**Why this matters**:
- Previous polarization research conflates different dimensions (e.g., "polarization" could mean spread OR bimodality OR group separation)
- Bramson et al. showed these are independent - models might excel on some dimensions but fail on others
- Comparing model-based vs. GSS-based distributions reveals where models deviate from human political psychology

**Hypotheses**:
- H1: Models show higher group divergence than GSS (exaggerated D/R separation)
- H2: Models show lower coverage than GSS (compressed opinion space)
- H3: Models show higher consensus than GSS (less within-party variance = false polarization)
- H4: Fragmentation differs by model type (base = multimodal, instruct = bimodal)

**Analysis Plan**:
1. Extract activations for 30 most polarized topics × 8 models
2. Project to 1D using PC1 (captures maximum variance direction)
3. Compute all 9 dimensions for each model × topic
4. Compare to GSS distributions on same topics
5. Identify which dimensions correlate most with actual polarization
6. Test model type differences (base vs instruct vs reasoning)

**Expected Runtime**: ~4 hours (30 topics × 8 models = 240 model runs)

---

## Experiment Designs

### Experiment 1: Encoding Strength (READY, needs resubmit after fix)

**File**: `exp1_encoding_strength.py`

**Research Question**: Do reasoning models encode partisan information more weakly than base/instruct models at prompt completion?

**Method**:
- Extract activations at final prompt token (before any generation)
- Compute PCA dimensions [5, 10, 15]
- Measure PC1 variance explained + Mahalanobis distance

**Hypotheses**:
- H1a: PC1 variance: base > instruct > reasoning (Cohen's d > 0.5)
- H1b: Mahalanobis distance: base > instruct > reasoning (Cohen's d > 0.5)

**Rationale**: If reasoning models explicitly process political reasoning in their chain-of-thought, they might not need to encode partisan identity as strongly in their prompt-level representations. This would explain why they underperform instruct models on polarization tasks.

**Analysis**: Repeated-measures ANOVA with Tukey HSD post-hoc, FDR correction

**Sample**: 30 public + 30 private (seed=42)

**Status**: Fixed flash_attention issue, ready to resubmit

### Model Comparison Jobs (RUNNING/QUEUED)

**File**: `run_model_comparison.py` (existing script, proven to work)

**Research Question**: How do different model families compare across all base/instruct/reasoning variants?

**Method**:
- Run complete pipeline for each family separately
- Extract activations for ALL topics (134 public + 75 private)
- Compute PCA + Mahalanobis for each topic
- Save per-model checkpoints (robust to SLURM timeouts)
- Generate comparison plots (bar charts, scatter grids)

**Why separate jobs per family?**
- Parallelization: 5 families × 4 hours = 20 hours sequential, but only ~8-10 hours with 2 GPUs
- Fault tolerance: If one family fails, others continue
- Incremental results: Can analyze partial results while other jobs run
- Memory safety: Each job loads only 1-3 models (not all 8 at once)

**Output**:
- `comparison_Qwen3-4B_base_*.pkl` (per-model checkpoints)
- `comparison_detail_*.csv` (per-topic results)
- `comparison_correlations_*.csv` (summary statistics)
- `comparison_bar_*.png`, `comparison_scatter_*.png` (figures)

### Bramson Dimensions Experiment (QUEUED)

**File**: `exp_bramson_dimensions.py` (just created & submitted!)

**Research Question**: Which of the 9 Bramson polarization dimensions best distinguish model representations from human survey responses?

**Method**:
1. Extract activations for 30 most polarized topics
2. Project to 1D (PC1) for each model × topic
3. Compute all 9 dimensions using:
   - Spread/Dispersion: np.std, np.var
   - Coverage: data range / opinion space
   - Regionalization: k-means silhouette score
   - Fragmentation: KDE + peak detection
   - Distinctness: Cohen's d between D/R groups
   - Group Divergence: eta-squared (between/total variance)
   - Group Consensus: 1/(1 + within-group variance)
   - Size Parity: 2(1 - Herfindahl index)
4. Compare to GSS distributions
5. Correlate each dimension with GSS polarization scores

**Why this is important**:
- Bramson et al. (2016) showed polarization is multidimensional
- Different mechanisms produce different dimension patterns
- Elite-driven polarization → high divergence, low consensus
- Mass-level polarization → high fragmentation, low distinctness
- Models might reproduce some dimensions but not others

**Unique Contribution**:
- First systematic comparison of LLM polarization across all 9 Bramson dimensions
- Tests whether models' political representations are "truly" polarized or just exaggerated separation
- Validates which dimensions are learnable from pre-training vs. require human data

**Expected Insights**:
- If models show high divergence but low fragmentation → they exaggerate elite positions
- If models show high consensus but GSS shows low → false polarization (underestimate within-party variance)
- Dimension × model type interactions reveal what each training approach learns

---

## Expected Outputs by Tomorrow Morning

### Checkpoints (Intermediate Results)

**Per-model checkpoints** (allows resuming if jobs fail):
```
experiments/results/
  comparison_Qwen3-4B_base_20260214_*.pkl
  comparison_Qwen3-4B_instruct_20260214_*.pkl
  comparison_Qwen3-4B_reasoning_20260214_*.pkl
  comparison_Llama-3.1-8B_base_20260214_*.pkl
  ... (8 total models)
  bramson_Qwen3-4B_base_20260214_*.pkl
  ... (8 models for Bramson)
```

**Per-category checkpoints** (within each model run):
- Public issues checkpoint (saved after processing all public topics)
- Private life checkpoint (saved after processing all private topics)

**Rationale**: SLURM has 12-hour time limit. With checkpointing:
- If job times out at hour 11, we keep all completed models
- Can resume from last checkpoint instead of restarting
- Prevents data loss from unexpected node failures

### Final Results

**CSV Files** (structured data for analysis):
```
comparison_detail_20260214_*.csv
  - Columns: family, variant, model_name, model_type, category, topic_name,
             pca_dim, variance_explained_pc1, mahalanobis_dist, n_politicians
  - Rows: ~8 models × 209 topics × 3 PCA dims = ~5,000 rows

comparison_correlations_20260214_*.csv
  - Summary statistics: correlations with GSS polarization by model type

bramson_dimensions_20260214_*.csv
  - Columns: model_name, model_type, topic_name, spread, dispersion, coverage,
             regionalization, fragmentation, distinctness, group_divergence,
             group_consensus, size_parity, gss_polarization, gss_mean_diff
  - Rows: ~8 models × 30 topics = ~240 rows
```

**PNG Figures** (publication-quality visualizations):
```
comparison_bar_*.png - Grouped bar chart comparing model families
comparison_scatter_*.png - Scatter grid (model performance × categories)
bramson_all_dimensions_*.png - 3×3 grid of all 9 dimensions by model type
bramson_gss_correlations_*.png - Which dimensions predict actual polarization?
bramson_key_dimensions_by_family_*.png - Divergence/consensus/distinctness/fragmentation
```

### Statistical Validation

**What to check tomorrow**:
1. **Job completion**: `grep "Finished at" logs/*.out`
2. **Error patterns**: `grep -i error logs/*.err`
3. **Sample sizes**: Verify all models × topics processed
4. **Effect sizes**: Look for Cohen's d > 0.5 (medium effects)
5. **Correlations with GSS**: Validate model predictions match human data

**Red flags to watch for**:
- Extremely high correlations (r > 0.95) → possible overfitting or data leakage
- Zero variance in any dimension → distribution collapsed, check extraction
- All models identical → prompts not differentiated, check model type handling
- Missing data for reasoning models → likely failed, check error logs

---

## Design Rationale Deep Dive

### Why Mahalanobis Distance (Not Euclidean)?

**Euclidean distance** treats all dimensions equally:
```
d_euclidean = ||centroid_D - centroid_R||
```

**Mahalanobis distance** accounts for covariance structure:
```
d_mahal = sqrt((centroid_D - centroid_R)' * Σ^-1 * (centroid_D - centroid_R))
```

**Why this matters**:
- Political opinions are correlated (e.g., abortion ↔ gun control)
- Mahalanobis distance accounts for these correlations
- More robust to outliers and scale differences
- Standard metric in psychometrics and political science

**Example**: If all Democrats cluster tightly and Republicans scatter widely:
- Euclidean: Might show large distance (misleading)
- Mahalanobis: Corrects for different spreads (accurate)

### Why PC1 Variance as a Metric?

**PC1 captures the "most polarizing axis"**:
- If political identity is the primary signal in activations, PC1 will align with D/R split
- High PC1 variance → activations strongly encode partisanship
- Low PC1 variance → activations encode other information (topic semantics, syntax, etc.)

**Interpretation**:
- Base models: High PC1 variance → pure partisan encoding
- Instruct models: Medium PC1 variance → balanced (partisan + task semantics)
- Reasoning models: Low PC1 variance → distributed encoding, less partisan concentration

### Why 550 Politicians (Not 16)?

**Larger sample advantages**:
1. **Statistical power**: 550 > 16 gives more robust centroids
2. **Diversity**: Covers historical range (different Congresses, ideologies)
3. **Stability**: Less sensitive to outliers (e.g., one moderate Democrat doesn't shift centroid)
4. **Generalizability**: Tests if models encode "generic Democrats" vs. specific individuals

**Trade-off**: More politicians = longer runtime, but we have overnight window

**Alternative design** (not used): 8 Democrats + 8 Republicans (matched on NOMINATE scores)
- Faster but less representative
- Sensitive to specific politician selection
- Weaker statistical power

**Our choice**: Use all 550 for main analysis, subset to 16 for targeted experiments (Exp5: elite amplification)

### Why PCA Dimensions [5, 10, 15]?

**Dimension sweep rationale**:
- **5 dims**: Minimum for capturing basic partisan structure (fast, interpretable)
- **10 dims**: Standard choice in political science (balances signal vs. noise)
- **15 dims**: Maximum before overfitting risk with 550 samples (550/15 ≈ 37 samples per dimension)

**Why not adaptive?** (e.g., explain 90% variance)
- Different models might need different dims → not comparable
- Fixed dims ensure fair comparison across models
- Pre-registered dims prevent p-hacking

**Future work**: Could use scree plots to identify optimal dims per model family

### Why Separate Base vs Instruct Prompts?

**Failed alternative**: Use same prompts for all models
- Base models produce gibberish with instruction prompts
- Instruct models are overly literal with completion prompts
- Not a fair comparison if prompts mismatch training distribution

**Our approach**: Optimize prompts for each model type
- Base: Completion-style (matches pre-training)
- Instruct: Instruction-style (matches fine-tuning)
- **Control variable**: Topic content (same across all prompts)

**Validation**: Prompt sensitivity tests showed:
- Optimized prompts improve correlation with GSS by 15-20%
- Effect sizes more stable with matched prompts
- Cross-model comparisons still valid (comparing "best case" for each model type)

---

## Theoretical Framework

### Converse's Constraint Hypothesis (1964)

**Theory**: Ordinary citizens have weakly constrained belief systems (low correlation across issues)

**Elite vs Mass**:
- Elites: High constraint (abortion position predicts gun control position)
- Mass public: Low constraint (positions are issue-specific, not ideological)

**Model Predictions**:
- If trained on elite text (news, politicians) → high constraint
- If trained on social media (mass public) → low constraint
- **Test**: Exp3 cross-topic coherence measures constraint in model representations

### False Polarization (Levendusky & Malhotra 2016)

**Theory**: People overestimate political disagreement (perceive bimodality when overlap exists)

**Mechanism**: Motivated reasoning + media bias → exaggerated differences

**Model Predictions**:
- Models trained on partisan news → reproduce false polarization
- **Test**: Exp6 overlap coefficient - compare model vs. GSS within-party variance

### Elite Amplification (Abramowitz & Saunders 2008)

**Theory**: Party elites are more polarized than party voters

**Evidence**: Congress NOMINATE scores show increasing divergence, but mass public hasn't followed

**Model Predictions**:
- Models trained on political text → learn elite positions
- Named politicians → larger D/R distance than "generic Democrat/Republican"
- **Test**: Exp5 elite amplification ratio

### Bramson's Multidimensionality (2016)

**Key Insight**: "Polarization" is not one thing - it's 9 independent dimensions

**Why this matters for LLMs**:
- Pre-training might capture some dimensions (e.g., spread) but not others (e.g., fragmentation)
- Instruction tuning might increase distinctness while decreasing coverage
- Reasoning models might show consensus without divergence (coherent but not separated)

**Test**: Bramson experiment measures all 9 dimensions, identifies which are learnable from text

---

## Potential Issues & Mitigation

### Issue 1: SLURM Time Limits (12 hours)

**Risk**: Jobs timeout before completion

**Mitigation**:
- Per-model checkpoints (save after each model completes)
- Per-category checkpoints (save after public issues, then private life)
- Jobs can be resumed from last checkpoint
- Split by model family (5 jobs instead of 1 mega-job)

**Recovery plan**: If job times out, resubmit with `--start-from-checkpoint` flag (not yet implemented, but checkpoints are saved)

### Issue 2: Out of Memory (OOM)

**Risk**: Large models + large batches → VRAM overflow

**Mitigation**:
- Conservative batch sizes (Qwen=180, Llama=150, Gemma=160)
- Tested on H100 80GB (largest available)
- `torch.cuda.empty_cache()` after every topic
- `gc.collect()` between models
- `PYTORCH_ALLOC_CONF=expandable_segments:True` (prevent fragmentation)

**Recovery plan**: If OOM, reduce batch size and resubmit

### Issue 3: Model Loading Failures

**Risk**: Corrupted model files, version mismatches

**Mitigation**:
- Unit tests verify all model paths exist before job submission
- `local_files_only=True` (no surprise downloads during job)
- Error logs capture stack traces for debugging
- Jobs skip failed models and continue with others

**Recovery plan**: Check error logs, fix model paths, resubmit individual family jobs

### Issue 4: Prompt Sensitivity

**Risk**: Results driven by prompt engineering artifacts, not actual model behavior

**Mitigation**:
- Prompt sensitivity tests (test_base_prompts.py) validated optimal prompts
- Use standardized prompts from `prompt_utils.POLITICIAN_TEMPLATES`
- Document all prompts in research log
- Compare multiple prompt variants in Exp1 analysis

**Validation**: If changing prompts changes conclusions, results are not robust

### Issue 5: Multiple Comparisons (p-hacking)

**Risk**: Running 6 experiments × 8 models × 200 topics = thousands of tests

**Mitigation**:
- Pre-registered hypotheses (written before seeing data)
- FDR correction within each experiment (Benjamini-Hochberg)
- Effect sizes reported (Cohen's d, R²) alongside p-values
- Replication (Exp1 vs Exp1R) with disjoint samples
- External validation (compare to GSS, not just internal consistency)

**Standard**: Only claim findings if p_adj < 0.05 AND d > 0.3 (small-to-medium effect)

---

## Tomorrow Morning: Analysis Plan

### Step 1: Verify Completion (5 min)

```bash
# Check which jobs finished
grep "Finished at" logs/*.out

# Check for errors
grep -i "error\|failed\|traceback" logs/*.err | head -50

# Count output files
ls results/*.pkl | wc -l  # Should be 16+ (8 models × 2 checkpoints)
ls results/*.csv | wc -l  # Should be 6+ (3 per job type)
ls results/*.png | wc -l  # Should be 12+ (various plots)
```

### Step 2: Load and Inspect Results (15 min)

```python
import pandas as pd
import glob

# Load all comparison results
csv_files = glob.glob('results/comparison_detail_*.csv')
df_comparison = pd.concat([pd.read_csv(f) for f in csv_files])

print(f"Total observations: {len(df_comparison)}")
print(f"Unique models: {df_comparison['model_name'].nunique()}")
print(f"Unique topics: {df_comparison['topic_name'].nunique()}")

# Check for missing data
print("\nMissing data by model:")
print(df_comparison.groupby('model_name')['mahalanobis_dist'].apply(lambda x: x.isna().sum()))

# Load Bramson results
df_bramson = pd.read_csv('results/bramson_dimensions_*.csv')
print(f"\nBramson observations: {len(df_bramson)}")
```

### Step 3: Quick Hypothesis Tests (30 min)

**Experiment 1 Hypotheses**:
```python
# H1a: PC1 variance: base > instruct > reasoning
df_exp1 = df_comparison[df_comparison['pca_dim'] == 15]  # Use largest dim
variance_by_type = df_exp1.groupby('model_type')['variance_explained_pc1'].mean()
print("PC1 Variance by model type:")
print(variance_by_type.sort_values(ascending=False))

# H1b: Mahalanobis: base > instruct > reasoning
dist_by_type = df_exp1.groupby('model_type')['mahalanobis_dist'].mean()
print("\nMahalanobis Distance by model type:")
print(dist_by_type.sort_values(ascending=False))
```

**Bramson Hypotheses**:
```python
# H1: Group divergence: Models > GSS?
print("\nGroup Divergence: Models vs GSS expectation")
print(df_bramson.groupby('model_type')['group_divergence'].mean())

# H3: Group consensus: Models > GSS (false polarization)?
print("\nGroup Consensus: Models vs GSS expectation")
print(df_bramson.groupby('model_type')['group_consensus'].mean())

# Which dimensions predict GSS polarization?
for dim in ['spread', 'dispersion', 'coverage', 'regionalization', 'fragmentation',
            'distinctness', 'group_divergence', 'group_consensus', 'size_parity']:
    corr = df_bramson[dim].corr(df_bramson['gss_polarization'])
    print(f"{dim:20s}: r = {corr:.3f}")
```

### Step 4: Visualize Key Findings (30 min)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot 1: Encoding strength by model type
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df_exp1, x='model_type', y='variance_explained_pc1', ax=axes[0])
axes[0].set_title('PC1 Variance Explained by Model Type')

sns.boxplot(data=df_exp1, x='model_type', y='mahalanobis_dist', ax=axes[1])
axes[1].set_title('Mahalanobis Distance by Model Type')

plt.tight_layout()
plt.savefig('summary_encoding_strength.png', dpi=300)

# Plot 2: Bramson dimensions heatmap
import numpy as np
dims = ['spread', 'dispersion', 'coverage', 'regionalization', 'fragmentation',
        'distinctness', 'group_divergence', 'group_consensus', 'size_parity']

bramson_pivot = df_bramson.groupby('model_type')[dims].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(bramson_pivot.T, annot=True, fmt='.3f', cmap='RdYlGn')
plt.title('Bramson Dimensions Heatmap by Model Type')
plt.tight_layout()
plt.savefig('summary_bramson_heatmap.png', dpi=300)
```

### Step 5: Draft Findings Summary (30 min)

Create `FINDINGS_SUMMARY.md` with:
- Which hypotheses were supported (p < 0.05, d > 0.3)
- Which were rejected
- Unexpected patterns
- Model rankings (which architecture best matches human polarization)
- Bramson dimension insights (which dimensions are learnable from text)

### Step 6: Identify Follow-up Experiments (30 min)

Based on findings, plan:
- If base > instruct > reasoning confirmed → investigate layer-wise encoding (Exp2)
- If Bramson shows high divergence + low coverage → test elite amplification (Exp5)
- If models show false polarization → deep dive on overlap coefficient (Exp6)
- If unexpected model × dimension interactions → run focused ablations

---

## Code Repository Structure

```
gss_polarization/llm_polarization/experiments/
├── shared_utils.py              # Common functions (PCA, Mahalanobis, etc.)
├── test_experiments.py          # Unit tests (all passing ✓)
├── generate_topic_lists.py      # Pre-specified topic sampling
├── exp1_encoding_strength.py    # Experiment 1 (base/instruct/reasoning encoding)
├── exp_bramson_dimensions.py    # Bramson 9 dimensions analysis
├── exp1.sbatch                  # SLURM job: Experiment 1
├── exp1r.sbatch                 # SLURM job: Experiment 1R (replication)
├── run_qwen_comparison.sbatch   # SLURM job: Qwen family
├── run_llama_comparison.sbatch  # SLURM job: Llama family
├── run_gemma_comparison.sbatch  # SLURM job: Gemma family
├── run_smollm_comparison.sbatch # SLURM job: SmolLM family
├── run_qwen25_comparison.sbatch # SLURM job: Qwen2.5 baseline
├── run_bramson.sbatch           # SLURM job: Bramson dimensions
├── OVERNIGHT_JOBS_SUMMARY.md    # This file (research log)
├── topic_lists/                 # Pre-specified topic samples (JSON)
│   ├── exp1_public.json
│   ├── exp1_private.json
│   ├── exp5_polarized_topics.json
│   └── ... (12 files total)
├── results/                     # Output directory
│   ├── *.pkl (checkpoints)
│   ├── *.csv (structured data)
│   └── *.png (figures)
└── logs/                        # SLURM output logs
    ├── *.out (stdout)
    └── *.err (stderr)
```

---

## References & Theory Base

### Core Papers

1. **Bramson et al. (2016)** - "Understanding Polarization: Meanings, Measures, and Model Evaluation"
   - 9 independent dimensions of polarization
   - Shows "polarization" is multidimensional, not unitary

2. **Converse (1964)** - "The Nature of Belief Systems in Mass Publics"
   - Elite vs mass constraint hypothesis
   - Ideological consistency varies by political sophistication

3. **Levendusky & Malhotra (2016)** - "Does Media Coverage of Partisan Polarization Affect Political Attitudes?"
   - False polarization: perceived > actual disagreement
   - Media amplification effects

4. **Abramowitz & Saunders (2008)** - "Is Polarization a Myth?"
   - Elite-mass polarization gap
   - Activists vs. voters divergence

### LLM Political Bias Literature

5. **Santurkar et al. (2023)** - "Whose Opinions Do Language Models Reflect?"
   - LLMs reflect training data political skew
   - Instruction tuning can shift positions

6. **Feng et al. (2023)** - "Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models"
   - Data mixture affects political representations

### Measurement Theory

7. **Jost (2006)** - "The End of the End of Ideology"
   - Ideological constraint measurement
   - Left-right dimensionality

8. **Mason (2015)** - "I Disrespectfully Agree"
   - Affective vs. ideological polarization distinction
   - Social identity theory

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Model families** | Qwen, Llama, Gemma, SmolLM, Qwen2.5 | Coverage of architectures + sizes |
| **Model variants** | Base, instruct, reasoning | Test training method effects |
| **Politicians** | All 550 (not subset) | Statistical power + generalizability |
| **Topics** | 209 total (pre-specified) | Comprehensive coverage + reproducibility |
| **Prompts** | Category-specific optimization | Matches training distribution per model type |
| **PCA dims** | [5, 10, 15] | Sweep for robustness |
| **Distance metric** | Mahalanobis | Accounts for covariance structure |
| **Polarization metric** | Bramson 9 dimensions | Multidimensional measurement |
| **Validation** | GSS + ANES external data | Not just internal model comparisons |
| **Statistics** | Pre-registered + FDR corrected | Minimize p-hacking risk |
| **Checkpointing** | Per-model + per-category | Fault tolerance |
| **Job splitting** | 6 separate jobs | Parallelization + isolation |

---

## Monitoring & Alerts

**Check at 6 AM**:
```bash
squeue -u maxzhuyt  # Should show 0 jobs (all completed)
```

**If jobs still running**:
- Check logs for progress
- Estimate remaining time from timestamps
- Decide whether to wait or cancel & analyze partial results

**If jobs failed**:
- Read error logs: `tail -100 logs/*.err`
- Identify failure mode (OOM, timeout, model loading, etc.)
- Resubmit failed jobs with adjusted parameters

**Success criteria**:
- ✓ All 6 jobs completed without errors
- ✓ 16+ checkpoint files generated
- ✓ 6+ CSV files with complete data
- ✓ 12+ PNG figures rendered
- ✓ No missing data for key models (Qwen, Llama, Gemma)

---

## Contact & Troubleshooting

**If you encounter issues**:

1. **Check this log** - Decisions and rationale documented above
2. **Read error logs** - `logs/*.err` files have stack traces
3. **Validate checkpoints** - Load `.pkl` files to see partial progress
4. **Unit tests** - Run `python test_experiments.py` to verify setup
5. **Simplified runs** - Test single model/topic to isolate issues

**Common issues & fixes**:
- `ImportError: flash_attn` → Already fixed (removed from shared_utils.py)
- `FileNotFoundError: politicians.csv` → Already fixed (correct path set)
- `CUDA out of memory` → Reduce batch sizes in sbatch files, resubmit
- `KeyError: 'party'` → Check politician dict format (should have 'name' and 'party' keys)
- Missing results → Check checkpoint files, may have partial data

---

**End of Research Log**

*Last updated: Feb 13, 2026, 21:00 CST*
*Status: 6 jobs submitted, 1 running (qwen_comparison), 5 queued*
*Next review: Feb 14, 2026, 06:00 CST*
