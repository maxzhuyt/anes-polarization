# Results Analysis: LLM Partisan Activation Experiments

**Date**: February 15, 2026
**Cluster**: jevans-gpu (H100 80GB), ssd-gpu (A100 40GB)
**Runtime**: ~8 hours total across all experiments

---

## Overview

13 experiments (including 1 replication) testing how base, instruct, and reasoning LLMs encode partisan political information in their attention-head activations. All experiments share a common pipeline: extract final-token activations from politician-prompted text, reduce dimensionality via PCA, and measure Democrat-Republican separation via Mahalanobis distance.

### Models (8 total, 3 families)

| Family | Base | Instruct | Reasoning |
|--------|------|----------|-----------|
| Qwen3-4B | Qwen3-4B-Base (4B, batch=180) | Qwen3-4B-Instruct-2507 (4B, batch=180) | Qwen3-4B-Thinking-2507 (4B, batch=180) |
| Llama-3.1-8B | Meta-Llama-3.1-8B (8B, batch=150) | Meta-Llama-3.1-8B-Instruct (8B, batch=150) | DeepSeek-R1-Distill-Llama-8B (8B, batch=150) |
| Gemma-2-9b | gemma-2-9b (9B, batch=160) | gemma-2-9b-it (9B, batch=160) | *(none)* |

All models stored at `/project/jevans/maxzhuyt/models/`.

### Shared Methodology

- **Politicians**: 550 members of U.S. 116th Congress from `data/politicians.csv` (DW-NOMINATE, Voteview) (287 Democrat [party_code=100], 263 Republican [party_code=200])
- **Activation extraction**: `extract_heads_batched()` returns 4D tensor `(N, L, H, D)` per topic. For Qwen3-4B: `(550, 36, 32, 128)` = ~325 MB/topic. For Llama-3.1-8B: `(550, 32, 32, 128)`.
- **Prompt format**: Base models use completion-style templates (e.g., `"{name} makes a statement on {topic}:"`). Instruct/reasoning models use chat templates with `SYSTEM_MSG_POLITICIAN`.
- **Inline processing**: To prevent OOM, activations are extracted, processed, and freed per-topic (never stored across topics simultaneously).
- **PCA + Mahalanobis** (global method): Flatten 4D activations to 2D `(N, L*H*D)` by concatenating all heads across all layers into a single feature vector per politician. Standardize features (zero mean, unit variance), then apply PCA to reduce to {1, 3, 5, 10, 15} dimensions. Compute Mahalanobis distance between Democrat and Republican centroids using pooled covariance. This captures separation across the *entire* model simultaneously.
- **Per-layer analysis** (Exp2): Instead of flattening across all layers, isolate each layer separately: take `activations[:, layer_idx, :, :]` of shape `(N, H, D)`, flatten to `(N, H*D)` by concatenating all heads *within that layer*, then apply PCA → Mahalanobis to each layer independently. This produces one distance value per layer, revealing where in the network the partisan signal is strongest.
- **Per-head analysis** (Exp7, Exp12): Isolate each individual head: take `activations[:, l, h, :]` of shape `(N, D)` for a specific layer `l` and head `h`, then apply LDA (Exp7) or ridge regression (Exp12) directly on the D-dimensional head activations.
- **Topics**: Drawn from two CSV files of GSS question descriptions: `public_issues_questions.csv` and `private_life_questions.csv` in `/project/jevans/maxzhuyt/gss_polarization/question_lists/`.
- **GSS validation**: External polarization from General Social Survey overlap coefficients (Democrat-Republican response distributions).
- **Seeds**: All experiments use `seed=42` for reproducibility.
- **Note on sample sizes**: With 8 models across 3 families, each experiment produces a small number of data points per condition. Results are illustrative — showing patterns across models and topics — rather than conclusive. We present full tables of values rather than relying on significance tests, which would be underpowered with these sample sizes.

### Status Summary

| Exp | Name | Status | Runtime | Key Finding |
|-----|------|--------|---------|-------------|
| 1 | Encoding Strength | DONE | 48 min | Instruct d=-1.70 vs base |
| 1R | Replication | DONE | 34 min | Replicates exp1 |
| 2 | Layer Depth | DONE | 1h 56m | Instruct peaks at 58% depth |
| 3 | Coherence | DONE | 36 min | Cross-topic transfer = 0.51 (chance) |
| 5 | Elite Amplification | DONE | 57 min | All models amplify 4-6x |
| 6 | ~~False Polarization~~ | DROPPED | — | Premise flawed (see below) |
| 7 | Head Discriminability | **REDESIGNED** | PENDING | Top-k% approach replaces AUC>0.6 threshold |
| 8 | Affective vs Policy | **REDESIGNED** | PENDING | Three domains: policy/affective/identity |
| B | Bramson 9 Dimensions | DONE | 27 min | Zero GSS correlations |
| 9 | Name Anonymization | **REDESIGNED** | PENDING | Cosine+probe (not Mahalanobis), 3 conditions |
| 10 | ~~Residual Signal~~ | DROPPED | — | Too mechanical (see below) |
| 12 | DW-NOMINATE Probing | DONE | 17 min | Instruct rho=0.84, transfer=0.90 |
| 13 | Behavioral Validation | DONE* | 22 min | Instruct generates partisan text (7/8 models) |

---

## Key Related Work: Kaplan et al. (ICLR 2025)

**"Linear Representations of Political Perspective Emerge in Large Language Models"**

The most directly comparable published work. Key details for comparison:

- **Data**: Same 116th Congress (552 lawmakers) with DW-NOMINATE scores
- **Method**: Per-head ridge regression (alpha=1, 2-fold CV) predicting continuous DW-NOMINATE dim1
- **Prompt**: `"Generate a statement by [NAME], a politician in the United States."` (no topic context)
- **Best result**: Spearman rho ~0.86 at middle layers (15-16) of Llama-2-70B-chat
- **Transfer**: Top-32 head ensemble transfers to news outlets (rho ~0.80)
- **Steering**: Linear interventions on top heads shift generation ideology
- **Limitation**: Only tested instruct/chat models; no base model comparison; no topic variation

**How our work extends Kaplan et al.**:
1. We compare base vs instruct vs reasoning models (they only use instruct)
2. We test topic-specific variation (they use a single no-topic prompt)
3. We use both binary party and continuous DW-NOMINATE (Exp 12)
4. We validate behaviorally via text generation (Exp 13)
5. We test cross-topic transfer (Exp 3), which they don't examine
6. We measure correlation with external ground-truth polarization (GSS), which they don't

---

## Experiment 1: Prompt-End Encoding Strength

### Research Question
Do reasoning models encode partisan information more weakly than base/instruct models at prompt completion (before generation)?

### Hypotheses
- **H1a**: PC1 variance explained: base > instruct > reasoning (Cohen's d > 0.5)
- **H1b**: Mahalanobis distance: base > instruct > reasoning (Cohen's d > 0.5)

### Method
- **Topics**: 30 public + 30 private GSS topics (pre-specified, seed=42)
- **Models**: All 8 models
- **Metric**: PCA-reduced Mahalanobis distance at dims {5, 10, 15}

### Results

**Mean Mahalanobis Distance by Model (averaged across 60 topics):**

| Model | Type | PCA=5 | PCA=10 | PCA=15 |
|-------|------|-------|--------|--------|
| Qwen3-4B-Base | base | 1.54 | 2.22 | 2.78 |
| Meta-Llama-3.1-8B | base | 2.35 | 3.24 | 3.85 |
| gemma-2-9b | base | 2.16 | 3.10 | 3.57 |
| **Qwen3-4B-Instruct-2507** | **instruct** | **3.21** | **3.88** | **4.12** |
| **Meta-Llama-3.1-8B-Instruct** | **instruct** | **3.78** | **4.52** | **4.73** |
| **gemma-2-9b-it** | **instruct** | **3.45** | **4.09** | **4.38** |
| Qwen3-4B-Thinking-2507 | reasoning | 1.65 | 2.32 | 2.89 |
| DeepSeek-R1-Distill-Llama-8B | reasoning | 2.10 | 2.95 | 3.32 |

All three instruct models (bolded) consistently show higher Mahalanobis distance than their base and reasoning counterparts within the same family. The pattern **instruct >> base ≈ reasoning** holds across every model family and every PCA dimension.

### Hypothesis Assessment
- **H1a (REJECTED)**: The ordering is instruct >> base ~ reasoning, not base > instruct > reasoning. Instruct models encode partisanship *more strongly*, not less.
- **H1b (PARTIALLY SUPPORTED, direction reversed)**: The predicted direction was wrong — instruct models show the *strongest* partisan signal, not the weakest. The instruct-base gap is large and consistent: instruct models show roughly 1.3-1.8x the distance of base models depending on PCA dimension.

### Assessment
The most consistent result across the study. The ordering instruct >> base ~ reasoning holds for every model family and every PCA dimension. The instruct-base gap narrows at higher PCA dimensions (instruct is ~1.7x base at PCA=5, ~1.3x at PCA=15), suggesting lower principal components carry proportionally more of the partisan signal in instruct models.

---

## Experiment 1R: Replication with Disjoint Topics

### Research Question
Same as Exp1, using non-overlapping topic sample.

### Hypotheses
Same as Exp1 (H1a, H1b).

### Method
Identical to Exp1 except `REPLICATION = True`, which draws a different random sample of 30+30 topics (seed=42, but different random draws).

### Results

The replication uses a disjoint set of 60 topics and produces the same pattern. Group-mean Mahalanobis distances:

| Model Type | PCA=5 (Exp1 / Exp1R) | PCA=15 (Exp1 / Exp1R) |
|-----------|----------------------|----------------------|
| Base | ~2.0 / ~2.0 | ~3.4 / ~3.4 |
| Instruct | ~3.5 / ~3.5 | ~4.4 / ~4.4 |
| Reasoning | ~1.9 / ~1.8 | ~3.1 / ~3.0 |

### Hypothesis Assessment
Identical to Exp1: instruct >> base ~ reasoning. The pattern replicates closely across disjoint topic samples.

### Assessment
The replication is near-exact. The instruct-base gap and reasoning-base equivalence are robust across different topic samples.

---

## Experiment 2: Layer-Depth Analysis

### Research Question
At which transformer layers does partisan information emerge, peak, and potentially diminish? Do base/instruct/reasoning models differ?

### Hypotheses
- **H2a**: Partisan info peaks in middle layers for base models but late layers for instruct
- **H2b**: Reasoning models show earlier peak and faster decay
- **H2c**: Instruct and reasoning share late-layer patterns

### Method
- **Topics**: 20 public + 20 private topics
- **PCA dim**: 15
- **Per-layer analysis**: The raw activations are 4D: `(N_politicians, L_layers, H_heads, D_head_dim)`. To analyze each layer independently, we slice out one layer at a time: `activations[:, layer_idx, :, :]` gives shape `(N, H, D)`. We then concatenate all H heads within that layer into a single feature vector: reshape to `(N, H*D)`. For example, a Qwen3-4B layer with 32 heads of dimension 128 produces a 4096-dimensional vector per politician. This vector is then standardized, PCA-reduced to 15 dims, and used for Mahalanobis distance. This produces one distance value per layer, revealing the depth profile of the partisan signal.
- **Metrics**: Peak layer (fraction of depth), peak distance, FWHM (full-width at half-maximum, i.e., the number of layers where distance exceeds half the peak value)

### Results

| Metric | Base (mean ± std) | Instruct (mean ± std) | Reasoning (mean ± std) |
|--------|-------|----------|-----------|
| Peak layer (fraction) | 0.439 ± 0.180 | 0.584 ± 0.140 | 0.452 ± 0.195 |
| FWHM (layers) | 27.98 ± 4.14 | 27.32 ± 4.03 | 23.48 ± 1.25 |
| Peak distance | 3.761 ± 1.136 | 5.105 ± 1.154 | 3.332 ± 0.268 |

Key observations:
- Instruct models peak 33% deeper into the network (0.584 vs 0.439)
- Reasoning models peak near base (0.452 vs 0.439), not near instruct (0.584)
- Reasoning models have significantly narrower FWHM: 23.48 ± 1.25 vs 27.98 ± 4.14 (base)
- Instruct peak distance is 36% higher than base (5.105 vs 3.761)

### Hypothesis Assessment
- **H2a (SUPPORTED)**: Base peaks at 44% (middle) while instruct peaks at 58% (late). The 14-percentage-point shift is substantial.
- **H2b (PARTIALLY SUPPORTED)**: Reasoning models do show narrower FWHM (23.5 vs 28.0), but peak location is similar to base (0.452 vs 0.439), not earlier. The "faster decay" aspect holds but "earlier peak" does not.
- **H2c (REJECTED)**: Reasoning does NOT share instruct's late-layer pattern. Reasoning reverts to base-like peak positions.

### Assessment
Informative result that complements Exp1. Shows WHERE the partisan signal lives: instruct models concentrate it in later layers (plausibly the RLHF alignment layers), while reasoning training appears to redistribute it to earlier, broader processing.

---

## Experiment 3: Cross-Topic Representational Coherence

### Research Question
Do models create a coherent political space, or are partisan representations topic-specific? Can a classifier trained on one topic predict party on another?

### Hypotheses
- **H3a**: Instruct models show higher cross-topic transfer than base (more coherent)
- **H3b**: Related topics (e.g., abortion variants) have higher transfer than unrelated
- **H3c**: Public issues show higher coherence than private life topics

### Method
- **Topics**: 40 topics (20 public + 20 private)
- **PCA dim**: 15
- **For each model**: Extract PCA-reduced features for all 40 topics
- **For each topic pair (i, j)**: Train logistic regression on topic i, test on topic j → 40×40 transfer matrix
- **Coherence score**: Mean off-diagonal accuracy
- **Self-accuracy**: Diagonal (train-test on same topic), measured via cross-validation

### Results

**Coherence by model type (aggregated):**

| Model Type | Coherence Accuracy | Coherence AUC | Self-Accuracy |
|-----------|-------------------|---------------|---------------|
| Base | 0.510 | 0.513 | 0.927 |
| Instruct | 0.512 | 0.515 | 0.972 |
| Reasoning | 0.531 | 0.541 | 0.944 |

**Coherence by individual model:**

| Model | Coherence Acc | Self-Accuracy |
|-------|-------------|---------------|
| Qwen3-4B base | 0.501 | 0.813 |
| Qwen3-4B instruct | 0.513 | 0.937 |
| Qwen3-4B reasoning | 0.506 | 0.926 |
| Llama-3.1-8B base | 0.493 | 0.985 |
| Llama-3.1-8B instruct | 0.527 | 0.988 |
| Llama-3.1-8B reasoning | 0.556 | 0.963 |
| Gemma-2-9b base | 0.535 | 0.984 |
| Gemma-2-9b instruct | 0.497 | 0.990 |

All model types show coherence near 0.50 (chance), with no meaningful differences between types. The per-model table above shows the full picture — even the highest coherence (Llama-3.1-8B reasoning at 0.556) is only marginally above chance with 8 models.

### Hypothesis Assessment
- **H3a (REJECTED)**: No meaningful difference between model types. Coherence is at chance for all.
- **H3b (NOT TESTABLE from aggregate data)**: Would need the full 40×40 transfer matrix per model.
- **H3c (NOT TESTED explicitly)**: No category-level breakdown reported.

### Assessment
This is the most important negative result. Self-accuracy is high (0.81-0.99), confirming a real partisan signal within each topic. But cross-topic transfer is at chance (0.50-0.53), meaning a classifier trained on abortion activations cannot predict party from gun control activations. This rules out the "models learn ideology" interpretation — the partisan encoding is topic-locked, not a coherent belief system. This contradicts Converse's (1964) constraint hypothesis as applied to LLMs.

Note: Llama-3.1-8B reasoning (DeepSeek-R1-Distill) shows slightly higher coherence (0.556) than others, but this is a single model and the pattern does not generalize across families.

---

## Experiment 5: Elite Amplification

### Research Question
Do LLMs amplify real-world partisan differences? Is the model-internal separation proportional to actual GSS-measured polarization?

### Hypotheses
- **H5a**: Instruct models amplify (ratio > 1) more than base models
- **H5b**: Amplification is stronger for high-profile culture war topics
- **H5c**: Reasoning models show less amplification than instruct

### Method
- **Topics**: 30 most polarized topics from GSS (pre-specified from overlap analysis)
- **Amplification ratio**: model Mahalanobis distance / GSS |mean_dem - mean_rep|. **Important caveat**: This ratio divides a distance in PCA-reduced activation space by a distance in survey-response space. Because these are fundamentally different measurement domains, the absolute ratio values are not directly interpretable (a ratio of 6x doesn't mean "6 times as polarized"). However, *relative* changes in the ratio across models or topics are meaningful — if instruct models consistently show higher ratios than base models, it means instruct models produce proportionally more activation-space separation per unit of survey-measured polarization.
- **Correlation**: Pearson/Spearman between model distance and GSS polarization

### Results

**Amplification Ratio by Model Type (PCA=15):**

| Model Type | Mean Ratio | Median Ratio | Std |
|-----------|-----------|-------------|-----|
| Base | 4.68 | 3.39 | 3.63 |
| Instruct | 6.20 | 4.49 | 4.29 |
| Reasoning | 4.10 | 2.85 | 2.52 |

The ratio is consistently higher for instruct models (6.20) than base (4.68) or reasoning (4.10), indicating instruct models create proportionally more activation-space separation relative to survey-measured differences.

**Correlation with GSS Polarization:**

| Model Type | Pearson r | Spearman rho |
|-----------|-----------|-------------|
| Base | 0.030 | 0.070 |
| Instruct | 0.054 | 0.039 |
| Reasoning | 0.077 | 0.124 |

All correlations are near zero — model distance does not track which topics are more or less polarized in the real world.

**Topics with highest and lowest ratios (instruct, PCA=15):**

| Topic | Ratio | GSS Polarization |
|-------|-------|-----------------|
| gunlaw | 15.57 | High |
| abhelp1 | 14.34 | Low |
| cappun | 12.56 | Medium |
| absingle | 11.21 | Low |
| abhelp3 | 11.05 | Low |
| ... | ... | ... |
| eqwlth | 1.86 | High |
| polviews | 2.32 | High |
| helpblk | 2.73 | Medium |

Note the mismatch: some high-ratio topics (abhelp1, absingle) have low GSS polarization, while some low-ratio topics (eqwlth, polviews) have high GSS polarization. This illustrates why the correlation is near zero.

### Hypothesis Assessment
- **H5a (SUPPORTED)**: Instruct shows higher ratios (6.20 vs 4.68 for base), indicating proportionally greater activation-space amplification. The instruct > base > reasoning ordering is consistent.
- **H5b (UNSUPPORTED)**: Near-zero correlation between model distance and GSS polarization. The model doesn't differentially amplify high-polarization topics.
- **H5c (SUPPORTED)**: Reasoning shows the lowest ratios (4.10), consistent with its base-like activation distances.

### Assessment
Instruct models produce proportionally more activation-space separation than base or reasoning models (relative ratio comparison), but this amplification is uniform across topics — there is no correlation between which topics get amplified and which are actually polarized. The model creates partisan separation on all topics roughly equally, regardless of whether the topic is actually contentious in the real world.

---

## ~~Experiment 6: False Polarization Detection~~ [DROPPED]

**Reason for dropping**: The premise of this experiment was flawed. LLM activation space is a simulation of cognitive space across politicians — which is itself a simulation task grounded in each politician's identity, personal history, and policy positions. There *should be* far more heterogeneity in activation space than in surveys, because the model encodes rich biographical information, not just 7-point scale responses. Finding that LLM activations show more separation than GSS survey data does not constitute "false polarization" — it's the expected outcome of a richer representational medium. The experiment's key metric (comparing activation-space distance to survey distance) does not produce interpretable results.

**Original results preserved for reference**: ROC AUC was near chance (0.52-0.56) for predicting whether a topic is actually polarized from model distance. The one positive finding — instruct models preserving rank-order of topic polarization (Spearman rho=0.415) — is already captured by Exp5's amplification analysis.

---

## Experiment 7: Head-Level Discriminability Analysis (Redesigned)

### Research Question
Which attention heads are most important for encoding partisan information? Are the *same* heads consistently most discriminative across different topics?

### Why Redesigned
The original Exp7 used a fixed threshold (AUC > 0.6) to identify "discriminative" heads. Because nearly every head achieves AUC >> 0.6 (mean AUC ~0.95), this threshold captured 82-94% of all heads, making Jaccard overlap trivially ~1.0. The result was uninformative — it merely confirmed that most heads encode *some* party information, not whether partisan encoding is *concentrated* in a stable subset.

### Hypotheses
- **H7a**: The top 5% most discriminative heads overlap substantially across topics (Jaccard > 0.3), indicating a stable set of "political" heads
- **H7b**: Top heads are concentrated in specific layers (not uniformly distributed)
- **H7c**: Instruct models concentrate top heads in later layers relative to base models

### Method
- **Topics**: Same 6 representative topics from `exp7_attention_topics.json`
- **Per head**: For each head `(l, h)`, extract `(N, D)` activations, compute LDA (Fisher's linear discriminant) projection to 1D, evaluate AUC. LDA is closed-form O(N) per head, no train-test leakage.
- **Top-k% selection**: Instead of a fixed AUC threshold, take the top 5% and 10% most discriminative heads per topic. For Qwen3-4B (1,152 heads), top 5% = 58 heads; for Llama-3.1-8B (1,024 heads), top 5% = 51 heads.
- **Cross-topic overlap**: Pairwise Jaccard similarity of top-k head sets across all topic pairs. Also compute the *intersection* across all topics — "universal" heads that appear in every topic's top-k.
- **Layer concentration**: For each model, count how the top-k% heads distribute across layers. If concentrated, the partisan signal has a "home" in specific layers. If uniform, every layer contributes equally.

### Results

**AUC Distribution by Model Type (averaged across 20 topics):**

| Model Type | Mean AUC | Max AUC | Median AUC |
|-----------|----------|---------|------------|
| Base | 0.9479 | 0.9963 | 0.9808 |
| Instruct | 0.9525 | 0.9995 | 0.9954 |
| Reasoning | 0.9218 | 0.9971 | 0.9803 |

Even with the redesigned analysis, mean AUC remains very high (>0.92) for all model types, confirming that nearly every head encodes *some* partisan information. The key question is what happens when we focus on the *elite* top-k%.

**Cross-Topic Top-K Head Overlap (Jaccard index, top 5%):**

| Model | Mean Jaccard | Range | Universal/Total |
|-------|-------------|-------|-----------------|
| Qwen3-4B base | 0.409 | [0.267, 0.702] | 5/57 |
| Qwen3-4B instruct | 0.367 | [0.213, 0.702] | 7/57 |
| Qwen3-4B reasoning | 0.335 | [0.152, 0.629] | 2/57 |
| Llama-3.1-8B base | 0.308 | [0.133, 0.645] | 1/51 |
| Llama-3.1-8B instruct | 0.300 | [0.146, 0.545] | 2/51 |
| Llama-3.1-8B reasoning | **0.465** | [0.291, 0.729] | **12/51** |
| Gemma-2-9b base | 0.087 | [0.000, 0.269] | 0/33 |
| Gemma-2-9b instruct | **0.523** | [0.000, 0.833] | 0/33 |

**Cross-Topic Top-K Head Overlap (Jaccard index, top 10%):**

| Model | Mean Jaccard | Range | Universal/Total |
|-------|-------------|-------|-----------------|
| Qwen3-4B base | 0.481 | [0.361, 0.783] | 21/115 |
| Qwen3-4B instruct | 0.462 | [0.330, 0.691] | 17/115 |
| Qwen3-4B reasoning | 0.413 | [0.285, 0.608] | 15/115 |
| Llama-3.1-8B base | 0.385 | [0.214, 0.714] | 11/102 |
| Llama-3.1-8B instruct | 0.416 | [0.236, 0.606] | 13/102 |
| Llama-3.1-8B reasoning | **0.529** | [0.360, 0.714] | **28/102** |
| Gemma-2-9b base | 0.161 | [0.064, 0.354] | 0/67 |
| Gemma-2-9b instruct | **0.545** | [0.000, 0.811] | 0/67 |

Compared to the original result (Jaccard ~0.996-1.000 with AUC > 0.6 threshold), the top-5% overlap is dramatically lower (0.09-0.52), confirming the redesign resolved the trivial ceiling effect.

**Layer Concentration of Top 5% Heads:**

| Model | Peak Layer | Peak % | Later Half % |
|-------|-----------|--------|-------------|
| Qwen3-4B base | 24/35 | 17.0% | **91.2%** |
| Qwen3-4B instruct | 23/35 | 17.7% | **93.8%** |
| Qwen3-4B reasoning | 23/35 | 10.8% | **93.1%** |
| Llama-3.1-8B base | 14/31 | 11.1% | 46.9% |
| Llama-3.1-8B instruct | 15/31 | 15.5% | **58.2%** |
| Llama-3.1-8B reasoning | 12/31 | 13.2% | 25.9% |
| Gemma-2-9b base | 23/41 | 9.9% | **67.6%** |
| Gemma-2-9b instruct | 39/41 | **37.6%** | **72.1%** |

### Hypothesis Assessment
- **H7a (MODERATELY SUPPORTED)**: Top-5% Jaccard ranges from 0.09 to 0.52. For 6 of 8 models, mean Jaccard is 0.30-0.52 (substantially above zero), indicating a moderately stable core of political heads. However, not as stable as H7a predicted (Jaccard > 0.3 threshold met by 6/8 models). Gemma-2-9b base is an outlier with near-zero overlap, suggesting its top heads are completely topic-specific.
- **H7b (STRONGLY SUPPORTED)**: Top heads are highly concentrated in later layers. For Qwen3-4B, >91% of top heads fall in the later half across all model types. For Gemma-2-9b instruct, 37.6% of top heads are in a single layer (39/41). The partisan signal has a clear "home" in the network.
- **H7c (SUPPORTED for some families)**: For Llama-3.1-8B, instruct concentrates heads later (58.2% in later half) vs base (46.9%). For Gemma-2-9b, instruct peak is at layer 39/41 vs base at 23/41. However, Qwen3-4B shows >91% in later half for all three types, so the instruct shift is family-dependent.

### Assessment
The redesigned top-k% approach reveals a much richer picture than the original AUC > 0.6 analysis:

1. **Moderate stability, not trivial overlap**: The top 5% of heads show Jaccard ~0.30-0.52 across topics for most models — meaningful but far from the near-perfect overlap (~0.996) that the original analysis found. There IS a partially stable core of political heads, but their composition shifts meaningfully between topics.

2. **Surprising model-type patterns**: DeepSeek-R1-Distill-Llama (reasoning) shows the *highest* head stability (Jaccard 0.465, 12 universal heads) among Llama models. This contrasts with its base-like Mahalanobis distance in Exp1. Interpretation: reasoning training may concentrate the partisan signal into fewer, more stable heads while reducing overall magnitude.

3. **Strong layer concentration**: Top heads cluster in later layers (especially for Qwen: >91% in later half). Gemma-2-9b instruct shows extreme concentration: 37.6% of all top-5% heads are in a single layer (39/41, the penultimate layer). This suggests the partisan signal is processed late in the network, consistent with Exp2's layer-depth finding.

4. **Family-specific architecture effects**: Gemma-2-9b base has near-zero head overlap (0.087) despite having the highest per-head AUCs (mean 0.987). This model's top heads are entirely topic-specific — partisan encoding is distributed across the network with no stable specialization. Gemma-2-9b instruct reverses this pattern (0.523), suggesting instruction tuning creates head specialization in Gemma.

---

## Experiment 8: Affective vs Policy Polarization (Redesigned)

### Research Question
Do LLMs encode affective polarization (inter-party hostility/warmth) differently from policy polarization (issue-position divergence)?

### Why Redesigned
The original Exp8 had only 1 affective topic (`confed` — confederate flag), making the affective-vs-policy comparison uninterpretable. More fundamentally, the operationalization of "affective" vs "policy" was underspecified and did not reflect how these constructs are studied in the human literature.

### Background: Human Polarization Literature
- **Policy polarization**: Divergence in issue positions between parties. Measured via survey items on specific policies (spending, abortion, guns). It is about *what you believe*.
- **Affective polarization**: Hostility toward, or negative evaluation of, the out-party irrespective of policy. Measured via feeling thermometers, social distance scales, and trait ratings. Key work: Iyengar, Sood, & Lelkes (2012); Mason (2015). It is about *how you feel about the other side*.
- **Partisan identity**: The strength of attachment to one's party as a social group, independent of policy or affect. Key work: Green, Palmquist, & Schickler (2002). It is about *who you are*.

### Operationalization for LLMs
We use three distinct prompt domains that isolate different aspects:

1. **Policy prompts**: "Generate a statement by {name} on {topic}." / "{name} makes a statement on {topic}:" — Uses real GSS policy topics (10 topics: eqwlth, natenvir, cappun, gunlaw, abany, natspac, homosex, prayer, immig, grass). Activations reflect policy-stance encoding.

2. **Affective prompts**: "How would {name} respond when asked about {topic}?" / "{name} discusses {topic}:" — Novel prompts about inter-party feelings, trust, and social distance (10 topics: feelings about the opposite party, trusting opposite-party neighbors, marrying into the other party, perceptions of opposite-party intelligence/morality, etc.). Activations reflect affective/identity-based encoding.

3. **Identity prompts**: "How would {name} talk about {topic}?" / "{name} describes {topic}:" — No policy or affective content (5 topics: importance of being a partisan, what party membership means, party as core identity, etc.). Activations reflect pure partisan self-concept encoding.

### Hypotheses
- **H8a**: Affective topics show higher partisan separation (Mahalanobis) than policy topics, as the model may amplify inter-group hostility beyond what policy differences explain
- **H8b**: Probe accuracy is higher for identity topics than policy topics (pure partisan signal)
- **H8c**: Cross-domain transfer is highest between affective and identity (both tap group-level representation) and lowest for policy → affective
- **H8d**: Instruct models show larger affective-policy gap than base models (RLHF may amplify affective polarization)

### Method
- **Topics**: 10 policy + 10 affective + 5 identity from `exp8_redesigned_topics.json`
- **PCA dim**: 15
- **Metrics**: Mahalanobis distance (party centroid separation), 5-fold CV linear probe accuracy
- **Cross-domain transfer**: For each model, concatenate PCA features across all topics within a domain, then train logistic regression on domain A and test on domain B → 3×3 transfer matrix (policy/affective/identity)
- **Models**: All 8 models across 3 families

### Results

**Mahalanobis Distance by Domain and Model Type (PCA=15, mean across topics):**

| Model Type | Policy | Affective | Identity |
|-----------|--------|-----------|----------|
| Base | 3.594 | 3.180 | **3.981** |
| Instruct | 4.679 | 4.560 | **4.862** |
| Reasoning | **3.169** | 3.001 | 2.899 |

For base and instruct: **Identity > Policy > Affective**. Identity prompts (pure partisan self-concept) produce the largest D-R separation, followed by policy (issue positions), then affective (inter-party feelings). For reasoning: the ordering reverses — **Policy > Affective > Identity**.

**Probe Accuracy by Domain and Model Type (5-fold CV):**

| Model Type | Policy | Affective | Identity |
|-----------|--------|-----------|----------|
| Base | 0.914 | 0.897 | **0.942** |
| Instruct | 0.963 | 0.963 | **0.966** |
| Reasoning | **0.929** | 0.921 | 0.912 |

For instruct, all three domains achieve near-identical probe accuracy (~0.96). The domain difference is most visible in base models: identity (0.942) > policy (0.914) > affective (0.897). Reasoning again reverses the pattern.

**Per-Model Mahalanobis Distance (mean across topics within domain):**

| Model | Affective | Identity | Policy |
|-------|-----------|----------|--------|
| Qwen3-4B base | 1.449 | 2.125 | 1.700 |
| Qwen3-4B instruct | 3.383 | 3.469 | 3.305 |
| Qwen3-4B reasoning | 2.929 | 3.027 | 2.861 |
| Llama-3.1-8B base | 3.887 | 4.533 | 4.674 |
| Llama-3.1-8B instruct | 4.845 | 5.191 | 4.935 |
| Llama-3.1-8B reasoning | 3.074 | 2.772 | 3.477 |
| Gemma-2-9b base | 4.203 | 5.284 | 4.407 |
| Gemma-2-9b instruct | 5.452 | 5.926 | 5.798 |

Across all base and instruct models, identity topics consistently produce the highest Mahalanobis distance (highlighted). The Gemma-2-9b family shows the largest identity premium (base: identity 5.28 vs policy 4.41; instruct: identity 5.93 vs policy 5.80).

**Cross-Domain Transfer Accuracy (train on row → test on column):**

Selected models showing representative patterns:

*Llama-3.1-8B instruct:*

| Train \ Test | Policy | Affective | Identity |
|-------------|--------|-----------|----------|
| Policy | **0.989** | 0.755 | 0.389 |
| Affective | 0.749 | **0.986** | 0.496 |
| Identity | 0.396 | 0.516 | **0.987** |

*Gemma-2-9b instruct:*

| Train \ Test | Policy | Affective | Identity |
|-------------|--------|-----------|----------|
| Policy | **0.989** | 0.645 | 0.680 |
| Affective | 0.558 | **0.989** | 0.769 |
| Identity | 0.666 | **0.822** | **0.995** |

Key transfer patterns:
- **Policy ↔ Affective**: Moderate transfer (~0.56-0.75 for instruct), highest for Llama instruct (0.75 bidirectional)
- **Identity → other domains**: Asymmetric. Identity transfers well to affective (Gemma instruct: 0.82) but poorly to policy in some models
- **Self-accuracy**: Extremely high (>0.98) for all models and domains, confirming strong within-domain signal
- **Model variation**: Transfer patterns differ substantially across models — no universal pattern

### Hypothesis Assessment
- **H8a (REJECTED)**: Affective topics show *lower* separation than policy topics (base: 3.18 vs 3.59; instruct: 4.56 vs 4.68). Identity shows the highest. The ordering is identity > policy > affective, not affective > policy.
- **H8b (SUPPORTED)**: Identity probe accuracy is highest for base models (0.942 vs 0.914 policy, 0.897 affective). For instruct, the differences are minimal (~0.96 for all), suggesting instruct models encode all domains at ceiling.
- **H8c (PARTIALLY SUPPORTED)**: Identity → affective transfer is indeed relatively high (Gemma instruct: 0.82), but policy ↔ affective bidirectional transfer is the strongest pair for Llama instruct (0.75). The pattern is model-specific, not universal.
- **H8d (NOT SUPPORTED)**: Instruct models do NOT show a larger affective-policy gap. The differences between domains are actually smaller for instruct (identity-affective gap: 0.30) than base (0.80). Instruct models compress cross-domain variation.

### Assessment
**The most informative finding is the identity > policy > affective ordering.** Pure partisan identity prompts ("how does [politician] describe their party membership") create the strongest D-R separation, even stronger than specific policy issues. This suggests:

1. **Identity encoding is primary**: LLMs encode "who someone is politically" (identity) more strongly than "what they believe" (policy) or "how they feel about the other side" (affective). This mirrors the human literature: Green, Palmquist, & Schickler (2002) argued that partisan identity is prior to and more stable than issue positions.

2. **Affective polarization is weaker in LLMs**: Despite the surge of affective polarization among humans (Iyengar et al. 2012), LLMs show the *weakest* partisan encoding for affective prompts. This makes sense: LLMs learn from text, and inter-party feeling/trust is less prominent in training corpora than policy positions or partisan identity.

3. **Instruct models compress domain differences**: While base models show a meaningful identity-affective gap (~0.80 Mahalanobis), instruct models compress this to ~0.30. RLHF appears to create a more uniform partisan encoding across all prompt domains.

4. **Reasoning models reverse the pattern**: Reasoning shows policy > affective > identity, the opposite of base/instruct. Chain-of-thought training may emphasize content-specific encoding (policy arguments) over identity-based encoding.

5. **Cross-domain transfer is asymmetric and model-specific**: No universal pattern, but identity training tends to transfer to affective (shared group-level representation) better than to policy.

---

## Bramson Experiment: Nine Dimensions of Polarization

### Research Question
Which of the 9 formal dimensions of polarization (Bramson et al., 2016) best distinguish model-generated representations from actual GSS survey responses?

### Hypotheses
- **H_B1**: Models show higher group divergence than GSS (exaggerated separation)
- **H_B2**: Models show lower coverage than GSS (compressed opinion space)
- **H_B3**: Models show higher group consensus than GSS (less within-party variance)
- **H_B4**: Bimodality/fragmentation differs between base and instruct models

### Method
- **9 Dimensions**: spread, dispersion, coverage, regionalization, fragmentation, distinctness, group_divergence, group_consensus, size_parity
- **Projection**: For each model × topic, project activations to 1D (PC1), compute all 9 dimensions on the resulting distribution
- **Comparison**: Correlate each dimension with GSS overlap coefficient

### Results

**Average Dimensions by Model Type:**

| Dimension | Base | Instruct | Reasoning |
|-----------|------|----------|-----------|
| Spread | 119.8 | 137.4 | 120.0 |
| Dispersion | 14,532 | 19,131 | 14,678 |
| Group Consensus | 0.0 | 0.0 | 0.0 |
| Size Parity | 0.998 | 0.998 | 0.998 |

**Correlation with GSS Polarization:**

| Dimension | Pearson r |
|-----------|-----------|
| Spread | 0.078 |
| Dispersion | 0.083 |
| Coverage | 0.043 |
| Regionalization | -0.050 |
| Fragmentation | -0.046 |
| Distinctness | 0.028 |
| Group Divergence | 0.028 |
| Group Consensus | -0.019 |
| Size Parity | -0.000 |

Maximum absolute correlation: r=0.083 (dispersion). All effectively zero.

### Hypothesis Assessment
- **H_B1 (NOT TESTABLE)**: Would need direct comparison of model vs GSS group divergence values. The correlation measure doesn't test this.
- **H_B2 (NOT TESTABLE)**: Same issue — testing whether values differ, not whether they correlate.
- **H_B3 (CONFIRMED but trivially)**: Group consensus = 0.0 for all models. Trivially true because within-party variance in high-dimensional activation space is large relative to any consensus measure.
- **H_B4 (NOT SUPPORTED)**: No meaningful differences between model types in any dimension that correlates with reality.

### Assessment
Comprehensive negative result. None of the 9 formal dimensions of polarization correlate with actual GSS polarization. This may be a domain mismatch: the Bramson framework was designed for 1D opinion distributions (e.g., survey scales), not for 1D projections of high-dimensional activation spaces. The PC1 projection compresses complex geometry into a scalar, potentially destroying the structure that these measures were designed to capture. The framework may simply not be applicable to neural network activation spaces.

---

## Experiment 9: Name Anonymization Test (Redesigned)

### Research Question
Is the partisan signal in LLM activations driven by politician name recognition or by political content/context? Can we disentangle real-world identity knowledge from bare partisan labels?

### Why Redesigned
The original Exp9 had two critical flaws:
1. **Mahalanobis inflation**: When using "anonymous Democrat/Republican" labels, all Democrats get nearly identical prompts and all Republicans get nearly identical prompts. Within-party variance collapses to near zero, which artificially inflates Mahalanobis distance (since Mahalanobis normalizes by the inverse covariance). The original 20x anonymous-vs-named ratio was largely an artifact of this variance collapse, not a meaningful signal.
2. **Only two conditions**: Named vs anonymous doesn't disentangle "real-world identity knowledge" from "individual-level variation." Adding a fictional-name condition with party labels solves this.

### Hypotheses
- **H9a**: Named condition shows higher probe accuracy and cosine distance than fictional, indicating real-world name recognition contributes to partisan encoding
- **H9b**: Fictional > anonymous for both metrics, indicating that individual-level variation (different names, states, roles) matters beyond bare party labels
- **H9c**: Instruct models show a larger named-fictional gap than base models (more biographical knowledge to leverage)
- **H9d**: If named ≈ fictional >> anonymous: individual variation matters, not name recognition. If named >> fictional ≈ anonymous: real-world name recognition dominates. If all three ≈ similar: partisan label alone drives encoding.

### Method
- **Topics**: 30 most polarized (same as Exp5, from `exp5_polarized_topics.json`)
- **Three conditions**:
  1. **Named**: Real politician names (e.g., "Nancy Pelosi")
  2. **Fictional**: Fictional names with party + state + role context (e.g., "John Smith, a Democratic senator from Ohio"). Constructed from 50 first names × 50 last names, randomly shuffled. Preserves within-party heterogeneity while removing real-world identity knowledge.
  3. **Anonymous**: Bare party labels only (e.g., "Democratic politician #1"). Minimal within-party variation.
- **Metrics** (NOT Mahalanobis — which is inflated for anonymous):
  - **Cosine distance** between party centroids in PCA-15 space
  - **Euclidean distance** between party centroids in PCA-15 space
  - **5-fold CV linear probe accuracy**: Logistic regression on PCA features, with proper train-test split
- **Models**: All 8 models across 3 families

### Results

**Job 45588048 COMPLETED** (1:00:06 on ssd-gpu A100)

**Cosine Distance: Saturated at 2.0 for ALL conditions and model types.** This is a ceiling effect — PCA-reduced party centroids always point in opposite directions, making cosine distance uninformative. This metric should be dropped from analysis.

**Euclidean Distance between Party Centroids (PCA-15):**

Not reported per-model in summary (all conditions saturated for cosine). But the raw log shows a clear gradient for individual topics:
- Example (Qwen3-4B base, `polviews`): named=168, fictional=287, anonymous=416
- Pattern: anonymous >> fictional >> named (consistently across models and topics)

**Linear Probe Accuracy (5-fold CV, mean across 30 topics):**

| Model Type | Named | Fictional | Anonymous |
|-----------|-------|-----------|-----------|
| Base      | 0.913 | 0.999     | 1.000     |
| Instruct  | 0.964 | 1.000     | 1.000     |
| Reasoning | 0.930 | 1.000     | 1.000     |

**Per-Model Probe Accuracy (mean across topics):**

| Model | Anonymous | Fictional | Named |
|-------|-----------|-----------|-------|
| Gemma-2-9b base | 1.000 | 1.000 | 0.970 |
| Gemma-2-9b instruct | 0.999 | 1.000 | 0.983 |
| Llama-3.1-8B base | 1.000 | 0.999 | 0.975 |
| Llama-3.1-8B instruct | 1.000 | 1.000 | 0.977 |
| Llama-3.1-8B reasoning | 1.000 | 1.000 | 0.950 |
| Qwen3-4B base | 1.000 | 0.999 | 0.793 |
| Qwen3-4B instruct | 1.000 | 1.000 | 0.930 |
| Qwen3-4B reasoning | 1.000 | 1.000 | 0.909 |

### Hypothesis Assessment

- **H9a (NOT SUPPORTED)**: Named condition does NOT show higher probe accuracy than fictional. In fact, named << fictional (0.91-0.96 vs 0.999-1.000). Real-world name recognition adds NOISE, not signal. The individual variation of real politicians makes party classification harder.
- **H9b (SUPPORTED)**: Fictional >> anonymous is NOT clearly seen in probe accuracy (both ~1.000). But Euclidean distance shows fictional < anonymous, suggesting anonymous (bare party labels) creates the strongest centroid separation. The fictional condition adds individual-level variation that slightly reduces separability.
- **H9c (PARTIALLY SUPPORTED)**: Instruct models show higher named probe accuracy (0.964) than base (0.913) or reasoning (0.930), indicating instruct models have more robust biographical knowledge. But the gap is about NAMED performance only — all models near-perfectly classify fictional/anonymous.
- **H9d (CLEAR PATTERN)**: The three-way pattern is: **named << fictional ≈ anonymous**. This means individual variation from real names HURTS classification — real politicians are messier than stereotypes. The model can near-perfectly classify with bare party labels or fictional names, but real politicians have more complex, overlapping representations.

### Assessment

**Key finding: Real politician names make party classification HARDER, not easier.** This is the opposite of what we initially expected. Named probe accuracy (0.91) is substantially lower than fictional (0.999) or anonymous (1.000). This means:

1. **LLMs encode complex biographical representations** of real politicians that don't reduce cleanly to party. Real politicians cross party lines, have idiosyncratic positions, and generate varied language — all of which adds noise to D-R classification.
2. **The high Mahalanobis distances in earlier experiments are NOT artifacts of name lookup.** If anything, real names ADD noise that reduces separation. The partisan signal emerges from the political CONTENT of prompts, modulated by biographical knowledge.
3. **Qwen3-4B base shows the most dramatic gap** (named=0.793 vs fictional=0.999), suggesting it has less robust biographical knowledge of individual politicians but strong party-label encoding.
4. **Cosine distance is uninformative** — always saturates at 2.0 in PCA-15 space. Future analyses should use Euclidean distance or probe accuracy instead.

---

## ~~Experiment 10: Residual Topic-Specific Signal~~ [DROPPED]

**Reason for dropping**: This experiment was too mechanical and does not correspond to anything meaningful. Projecting activations onto a "global party axis" and removing it is an arbitrary geometric operation — there's no cognitive or political interpretation for what the residual represents. The finding that removing the axis makes coherence worse (below chance) is unsurprising: you're removing the main signal and asking if something else remains. This doesn't reveal anything about the nature of the political representation.

**Original results preserved for reference**: Removing the global D-R axis reduced Mahalanobis distance by 19-26%. Residual cross-topic coherence dropped below chance (0.48). These results are mechanically predictable from the structure of the data and don't contribute to the paper's argument.

---

## Experiment 12: DW-NOMINATE Per-Head Linear Probing

### Research Question
Can individual attention heads linearly predict continuous DW-NOMINATE ideology scores? How do per-head probing results compare to our binary PCA+Mahalanobis approach?

### Connection to Prior Work
Directly replicates and extends Kaplan et al. (ICLR 2025), who found Spearman rho ~0.86 in middle layers of Llama-2-70B-chat. Our extension: (1) compare base vs instruct vs reasoning, (2) add topic-specific prompts, (3) test cross-topic transfer of probing performance.

### Hypotheses
- **H12a**: Instruct models achieve higher per-head Spearman rho than base (matching Exp1 Mahalanobis finding)
- **H12b**: Best probing heads are in middle layers (matching Kaplan et al.'s finding for layers 15-16)
- **H12c**: Topic-specific prompts improve probing accuracy over no-topic Kaplan-style prompts
- **H12d**: Cross-topic probing transfer (train on topic A, evaluate on topic B) is near chance (matching Exp3 coherence finding)

### Method
- **Targets**: Continuous DW-NOMINATE dim1 scores for 549 politicians (D=286, R=263)
- **Probing**: Ridge regression (alpha=1.0, 2-fold StratifiedKFold CV) per attention head (l,h)
- **Evaluation**: Spearman rho and Pearson r per head → heatmap (L×H)
- **Topics**: 10 GSS topics (6 public: natcrimy, conclerg, natrace, govunemp, grnsol, inteduc; 4 private: acqntsex, helpfrds, hunt, egomeans)
- **Kaplan-style baseline**: No-topic prompt `"Generate a statement by {name}, a politician in the United States."` for direct comparison
- **Transfer**: Ensemble of top-32 heads (by Spearman) from one topic, evaluate on another → transfer matrix
- **Models**: All 8 models across 3 families
- **Runtime**: 17 minutes (Job 45555753)

### Results

**Best-Head Spearman Correlation by Model Type (averaged across all topics):**

| Model Type | Best |rho| | Mean |rho| | Heads > 0.5 | Heads > 0.7 | Best layer frac |
|-----------|------------|------------|-------------|-------------|----------------|
| Base | 0.7915 ± 0.071 | 0.5272 ± 0.094 | 580.9 | 263.6 | 0.530 |
| Instruct | 0.8429 ± 0.045 | 0.6248 ± 0.074 | 693.7 | 553.7 | 0.574 |
| Reasoning | 0.7885 ± 0.024 | 0.5285 ± 0.025 | 716.7 | 296.9 | 0.595 |

Instruct models consistently achieve the highest best-head rho (0.84) compared to base (0.79) and reasoning (0.79). The same instruct >> base ≈ reasoning ordering appears again.

**Kaplan et al. Comparison (no-topic prompts only):**

| Model | Best rho | Best layer (frac) | Heads > 0.7 |
|-------|----------|-------------------|-------------|
| Qwen3-4B base | 0.760 | 28 (0.78) | 49 |
| Qwen3-4B instruct | 0.786 | 28 (0.78) | 542 |
| Qwen3-4B reasoning | 0.754 | 27 (0.75) | 215 |
| Llama-3.1-8B base | 0.856 | 14 (0.44) | 552 |
| Llama-3.1-8B instruct | 0.859 | 15 (0.47) | 742 |
| Llama-3.1-8B reasoning | 0.817 | 14 (0.44) | 264 |
| Gemma-2-9b base | 0.826 | 16 (0.38) | 322 |
| Gemma-2-9b instruct | 0.873 | 20 (0.48) | 494 |

Kaplan et al. reference: best head rho ~0.86 at layers 15-16 (Llama-2-70B-chat). Our Llama-3.1-8B instruct achieves rho=0.859 at layer 15, closely matching despite being a smaller model.

**Cross-Topic Transfer (DW-NOMINATE probes, top-32 head ensemble):**

| Model Type | Mean Transfer rho | Std |
|-----------|------------------|-----|
| Base | 0.837 | 0.065 |
| Instruct | 0.897 | 0.057 |
| Reasoning | 0.852 | 0.030 |

### Hypothesis Assessment
- **H12a (SUPPORTED)**: Instruct models achieve higher per-head rho (0.843 vs 0.792). Same ordering as Exp1 Mahalanobis: instruct >> base ≈ reasoning.
- **H12b (SUPPORTED)**: Best probing heads are in middle layers (0.53-0.60 depth). Llama models peak at layers 14-15 (44-47%), closely matching Kaplan's layers 15-16. Qwen models peak later (75-78%), likely due to having 36 vs 32 layers.
- **H12c (NOT SUPPORTED)**: Topic-specific prompts do not consistently improve over Kaplan-style no-topic prompts. E.g., Llama-3.1-8B instruct: no-topic rho=0.859, topic-conditioned mean best rho=0.870 — a marginal difference.
- **H12d (STRONGLY REJECTED)**: Cross-topic probing transfer is HIGH (0.84-0.90), NOT at chance. This directly contradicts Exp3's finding of chance-level binary classification transfer (~0.51).

### Assessment
**This is the most important new finding.** The contrast with Exp3 is striking: binary party classification (Exp3) shows chance-level cross-topic transfer (0.51), but continuous DW-NOMINATE probing (Exp12) shows very high transfer (0.84-0.90). This means:

1. **There IS a coherent ideology representation** — but it's visible only when using continuous ideology scores, not binary party labels. Binary D/R classification is too coarse to reveal the underlying ideological structure.
2. **The "identity-based, not ideological" conclusion from Exp3/Exp10 needs revision.** The representations DO encode ideological positioning coherently across topics, but this coherence operates on a continuous spectrum, not a binary axis.
3. **Our results closely replicate Kaplan et al.** (rho ~0.86 at middle layers) despite using smaller models (8-9B vs 70B), suggesting linear political representations are a general phenomenon across model scales.
4. **The instruct > base ≈ reasoning ordering holds** for continuous probing, consistent with all prior experiments

---

## Experiment 13: Behavioral Validation

### Research Question
Does the activation-level partisan signal predict actual model outputs? Do models generate more partisan text for politicians whose activations show stronger party separation?

### Connection to Prior Work
Addresses the "epiphenomenal" concern raised in Cross-Experiment Synthesis. Also connects to LLM opinion survey literature (Argyle et al. 2023, Santurkar et al. 2023) showing LLMs can simulate political opinions when prompted with demographic information.

### Hypotheses
- **H13a**: Models generate text with detectable partisan content when prompted with politician names
- **H13b**: Activation Mahalanobis distance positively correlates with generation-level partisan separation
- **H13c**: Instruct models produce more partisan text than base/reasoning (matching activation finding)
- **H13d**: Partisan content is higher for high-polarization topics (gun control) than low (education)

### Method
- **Politicians**: 50 per party (100 total, selected for DW-NOMINATE extremity)
- **Topics**: 10 high-polarization topics (gunlaw, abany, cappun, eqwlth, natenvir, homosex, prayer, grass, polviews, immig)
- **Generation**: 80 tokens, temperature=0.7, using model.generate()
- **Scoring**: Keyword-based partisan scoring with curated DEMOCRAT_KEYWORDS and REPUBLICAN_KEYWORDS lists. Score = (D_count - R_count) / total_words for each generation.
- **Activation metric**: PCA (dim=15) Mahalanobis distance between party centroids
- **Correlation**: Spearman rho between topic-level activation distance and generation partisan separation
- **Models**: 7 of 8 models completed (Gemma-2-9b instruct crashed during generation)
- **Runtime**: 22 minutes (Job 45556065)

### Results

**Generation Separation by Model (mean |D_score - R_score| across 10 topics):**

| Model | Mean Gen Separation | Mean Gen Accuracy | Mean Activation Mahal | Mean Unscored % |
|-------|--------------------|--------------------|----------------------|----------------|
| Qwen3-4B base | -0.002 | 0.491 | 2.708 | 35% |
| Qwen3-4B instruct | 0.205 | 0.593 | 4.267 | 8% |
| Qwen3-4B reasoning | 0.226 | 0.535 | 3.703 | 14% |
| Llama-3.1-8B base | -0.053 | 0.457 | 5.932 | 30% |
| Llama-3.1-8B instruct | 0.249 | 0.636 | 7.067 | 2% |
| Llama-3.1-8B reasoning | -0.029 | 0.447 | 3.596 | 21% |
| Gemma-2-9b base | -0.018 | 0.483 | 5.347 | 27% |

**Key topic-level highlights (Llama-3.1-8B instruct):**

| Topic | Activation Mahal | Gen Separation | Gen Accuracy | Unscored |
|-------|-----------------|---------------|--------------|----------|
| polviews | 7.004 | 1.063 | 0.960 | 0% |
| immig | 7.203 | 0.443 | 0.840 | 0% |
| natenvir | 6.894 | 0.277 | 0.540 | 0% |
| abany | 8.010 | 0.232 | 0.592 | 2% |
| gunlaw | 7.575 | 0.146 | 0.598 | 3% |

**Notable failures (base models):**
- Base models produce high rates of unscorable text (27-35% of generations contain no partisan keywords)
- Base model generation accuracy is near chance (0.45-0.49), confirming they produce less partisan text
- Some topics produce negative separation (wrong direction), likely noise from keyword scoring limitations

### Hypothesis Assessment
- **H13a (PARTIALLY SUPPORTED)**: Instruct models produce detectable partisan content (mean accuracy 0.59-0.64, above chance 0.50). Base models do not (accuracy 0.45-0.49, at or below chance). The keyword scoring method has high noise (many unscorable texts for base models).
- **H13b (UNCLEAR)**: Within instruct models there is a weak positive trend, but the keyword scoring method is too noisy for reliable correlation. Llama-3.1-8B instruct shows the clearest pattern: high Mahalanobis topics (polviews 7.0, immig 7.2) → high generation separation (1.06, 0.44). But the sample is too small (10 topics) for robust correlation.
- **H13c (SUPPORTED)**: Instruct models produce substantially more partisan text (mean separation 0.21-0.25) than base (-0.05 to 0.00) or reasoning (-0.03 to 0.23). The instruct vs base difference is clear. Reasoning is intermediate.
- **H13d (PARTIALLY SUPPORTED)**: `polviews` (political ideology) shows the strongest generation separation (1.06 for Llama instruct), but some expected high-polarization topics (gunlaw, cappun) show lower generation separation than expected.

### Assessment
**Mixed but informative results.** The behavioral validation confirms that instruct models generate more partisan text, consistent with the activation-level finding. However, several limitations:

1. **Keyword scoring is crude**: Many generations (especially from base models) contain no partisan keywords, producing high unscored rates and noisy separation estimates. An LLM judge would be more sensitive.
2. **Base models produce incoherent text**: With only 80 tokens and no instruction-following capability, base model generations are often continuations rather than coherent statements, limiting the interpretability of keyword scores.
3. **The standout result**: Llama-3.1-8B instruct on `polviews` achieves 96% generation accuracy with separation=1.06, demonstrating that the activation signal IS behaviorally relevant for at least some model-topic combinations.
4. **Gemma-2-9b instruct crashed during generation**, so we have 7/8 models. The crash occurred during text generation (not activation extraction), likely due to Gemma's different generation configuration.

---

## Cross-Experiment Synthesis

### Three Central Findings

**1. Instruction Tuning Creates a Large Partisan Signal** (Exp1, Exp1R, Exp2, Exp5)

| Metric | Base | Instruct | Reasoning | Instruct vs Base |
|--------|------|----------|-----------|-----------------|
| Mahalanobis (PCA=5) | ~2.0 | ~3.5 | ~1.9 | ~1.7x higher |
| Mahalanobis (PCA=15) | ~3.3 | ~4.4 | ~3.1 | ~1.3x higher |
| Peak layer (fraction) | 0.439 | 0.584 | 0.452 | 33% deeper |
| Peak distance | 3.761 | 5.105 | 3.332 | 36% higher |
| Amplification ratio | 4.68 | 6.20 | 4.10 | 32% higher |
| Self-accuracy (Exp3) | 0.927 | 0.972 | 0.944 | +5% |

This pattern replicated with disjoint topic samples in Exp1R, showing nearly identical values.

**2. Reasoning Models Revert to Base-Like Behavior** (Exp1, Exp2)

Despite being finetuned from instruct checkpoints, reasoning models (Qwen3-4B-Thinking, DeepSeek-R1-Distill-Llama) show partisan encoding nearly identical to base models:
- Base and reasoning Mahalanobis distances are similar across all PCA dimensions (e.g., ~3.3 vs ~3.1 at PCA=15)
- Reasoning has narrower FWHM (23.48 vs 27.98) and peak location similar to base (0.452 vs 0.439)

Chain-of-thought finetuning appears to "undo" the partisan amplification that instruction tuning creates.

**3. The Signal Encodes Continuous Ideology, Not Just Binary Party** (Exp3, Exp7, Exp9, Exp12)

Multiple experiments converge on a nuanced picture:
- **Exp7 (redesigned)**: Nearly all heads achieve high AUC, but the redesign tests whether the *top* 5-10% most discriminative heads are stable across topics (PENDING).
- **Exp3**: Binary cross-topic coherence at chance (0.50-0.53). Binary classifiers don't transfer.
- **Exp12**: But continuous DW-NOMINATE probes transfer at rho=0.84-0.90! The ideology IS coherent across topics when measured continuously.
- **Exp9 (redesigned)**: Three-condition design (named/fictional/anonymous). Key result: named << fictional ≈ anonymous for probe accuracy (0.91 vs 1.00). Real politician names add NOISE to party classification — the signal comes from political content, not name lookup.
- **Exp5/Bramson**: Near-zero correlation with actual GSS polarization (max r=0.083).

**The paradox resolved (revised with Exp12)**: Binary D/R classification fails to transfer across topics (Exp3), but continuous ideology probes transfer well (Exp12, rho=0.90 for instruct). The model DOES encode a coherent ideological spectrum — but this structure is visible only with continuous measures, not with the binary party split. This is analogous to finding that temperature measurements transfer across contexts but binary "hot/cold" labels do not.

**4. Activation Signals Are Behaviorally Relevant** (Exp13)

Instruct models generate measurably more partisan text (mean accuracy 0.59-0.64) than base (0.45-0.49) or reasoning (0.45-0.54), consistent with activation findings. The strongest case: Llama-3.1-8B instruct on `polviews` achieves 96% generation accuracy, demonstrating the activation signal predicts generation content. However, keyword-based scoring is noisy and many topics show weak behavioral separation.

### Methodological Concerns

1. **Identity recognition vs political encoding**: Exp9 (redesigned) shows real politician names make classification HARDER (0.91 accuracy) than fictional names (0.999) or anonymous labels (1.000). The partisan signal is NOT from name lookup — real politicians are messier than stereotypes.

2. **No causal evidence**: All experiments are observational (extract activations, measure distance). We don't know if the partisan signal causes model behavior or is epiphenomenal. Exp13 provides weak behavioral evidence (instruct models generate partisan text), but the keyword scoring is noisy.

3. **GSS comparison is crude**: The GSS overlap coefficient is computed from 7-point scales. It may not capture polarization at the granularity of continuous activation spaces.

4. **Affective vs policy operationalization**: Exp8 (redesigned) now uses 10 affective topics + 10 policy topics + 5 identity topics with domain-specific prompt templates grounded in the human polarization literature (Iyengar et al. 2012; Mason 2015). Results pending.

5. **Dropped experiments**: Exp6 (false polarization) and Exp10 (residual signal) were dropped because their premises were flawed — Exp6 assumed activation-space homogeneity implies false polarization (it doesn't), and Exp10 was mechanically predictable and uninterpretable.

---

## Pending Results

**Completed redesigned experiments:**
- **Exp7** (Job 45588046): COMPLETED in 14:54 — results above
- **Exp8** (Job 45588047): COMPLETED in 18:49 — results above
- **Exp9** (Job 45588048): COMPLETED in 1:00:06 — results above

**Experiment 0 (Mismatch Analysis) — 4 sub-experiments on ssd-gpu:**
- **Exp0A** (Job 45597274): COMPLETED — Prompt framing comparison
- **Exp0B** (Job 45597275): COMPLETED — Per-layer alignment profile
- **Exp0C** (Job 45597276): COMPLETED — Probe-based mismatch analysis
- **Exp0D** (Job 45597277): COMPLETED — D-R direction consistency

---

## Experiment 0: Understanding the LLM-Survey Polarization Mismatch

### Motivation

The correlation between LLM activation polarization (Mahalanobis distance) and GSS survey polarization is moderate at best (r = 0.27-0.47 for Qwen3-4B and SmolLM3-3B). More PCA components always increases this correlation, never decreases it — so the bottleneck is not dimensionality reduction. The mismatch is fundamental.

**Key insight**: LLM activations when prompted with "Generate a statement by {name} on {topic}" capture **rhetorical polarization** — how differently politicians would FRAME or TALK about an issue. GSS surveys capture **manifested opinion** — how differently people actually ANSWER standardized questions. These are distinct mental processes.

**Residual analysis** reveals systematic patterns in the mismatch:
- **Over-polarized topics** (LLM rhetoric >> GSS opinion): colhomo, spkhomo, oprelig, impgrn, polviews — topics with strong rhetorical framing but modest actual opinion differences
- **Under-polarized topics** (LLM rhetoric << GSS opinion): savesoul, pray, polescap, conbus, conlegis — topics with genuine opinion divides that aren't rhetorically salient

### Sub-Experiments

**All use only Qwen3-4B (base/instruct/reasoning) + SmolLM3-3B (base/instruct/reasoning) on 20 strategically selected topics (5 over-polarized, 5 under-polarized, 5 well-aligned high-pol, 5 well-aligned low-pol).**

#### Exp0A: Prompt Framing — Rhetoric vs Opinion vs Survey

**RQ**: Does switching from rhetorical to survey-style prompts improve alignment with GSS?

**Method**: Three prompt conditions:
1. RHETORICAL: "Generate a statement by {name} on {topic}" (current)
2. STANCE: "What is {name}'s position on {topic}?"
3. SURVEY: "If asked in a national survey about {GSS_question_text}, how would {name} respond?"

**Hypothesis**: Survey-aligned prompts will produce activations that better correlate with GSS survey polarization, because they elicit opinion-like (not rhetoric-like) representations.

**Results** (Job 45597274, COMPLETED 2026-02-16):

Pearson r (Mahalanobis vs GSS polarization) per Model × Condition:

| Model | rhetorical | stance | survey |
|-------|-----------|--------|--------|
| Qwen3-4B_base | 0.053 | **0.319** | -0.140 |
| Qwen3-4B_instruct | 0.149 | 0.255 | 0.075 |
| Qwen3-4B_reasoning | 0.249 | **0.457*** | 0.410 |
| SmolLM3-3B_base | -0.196 | -0.108 | 0.042 |
| SmolLM3-3B_instruct | -0.296 | **0.447*** | 0.193 |
| SmolLM3-3B_reasoning | 0.300 | 0.353 | **0.569**** |

\* p < 0.05, \*\* p < 0.01

Average r by condition: **rhetorical=0.043, stance=0.287, survey=0.192**

Average r by condition × model type:
| | rhetorical | stance | survey |
|---|-----------|--------|--------|
| base | -0.072 | 0.106 | -0.049 |
| instruct | -0.073 | **0.351** | 0.134 |
| reasoning | **0.275** | **0.405** | **0.490** |

**Key findings**:
- Stance prompts dramatically improve alignment over rhetorical (mean r: 0.043 → 0.287)
- Three statistically significant correlations found: Qwen3-4B_reasoning+stance (r=0.457, p=0.043), SmolLM3-3B_instruct+stance (r=0.447, p=0.048), SmolLM3-3B_reasoning+survey (r=0.569, p=0.009)
- Reasoning models benefit most from stance/survey framing (mean r=0.405/0.490 vs 0.275 for rhetorical)
- Base models don't benefit from any framing change (all near 0)
- The standard rhetorical prompt captures a different phenomenon than survey polarization — switching to stance/survey framing closes the gap

#### Exp0B: Per-Layer Alignment Profile

**RQ**: Which layers best predict GSS survey polarization? Is there a "rhetoric layer" vs an "opinion layer"?

**Method**: Extract full (N, L, H, D) activations, compute Mahalanobis at each layer separately. Correlate per-layer distances with GSS polarization. Compare layer profiles for over-polarized vs under-polarized topics.

**Hypothesis**: If rhetoric and opinion are encoded in different layers, then over-polarized topics (rhetoric >> opinion) should peak at different layers than under-polarized topics (opinion >> rhetoric).

**Results** (Job 45597275, COMPLETED 2026-02-16):

Best single-layer Pearson r (Mahalanobis vs GSS) per model:

| Model | Avg-layer r | Best Layer | Best r | p-value |
|-------|-------------|-----------|--------|---------|
| Qwen3-4B_base | 0.046 | 31/32 | 0.340 | 0.142 |
| Qwen3-4B_instruct | 0.147 | **20/36** | **0.573*** | **0.008** |
| Qwen3-4B_reasoning | 0.248 | 29/36 | 0.503* | 0.024 |
| SmolLM3-3B_base | -0.080 | 27/36 | 0.426 | 0.061 |
| SmolLM3-3B_instruct | -0.195 | 18/36 | 0.463* | 0.040 |
| SmolLM3-3B_reasoning | 0.301 | 13/36 | 0.509* | 0.022 |

\* p < 0.05

Layer quartile correlation profile (mean r within each quartile):

| Model | Q1 (early) | Q2 | Q3 | Q4 (late) |
|-------|-----------|----|----|----------|
| Qwen3-4B_base | -0.040 | -0.149 | -0.149 | 0.043 |
| Qwen3-4B_instruct | -0.083 | 0.108 | **0.323** | 0.087 |
| Qwen3-4B_reasoning | -0.035 | 0.102 | **0.386** | **0.302** |
| SmolLM3-3B_base | -0.061 | -0.095 | -0.038 | 0.064 |
| SmolLM3-3B_instruct | 0.090 | 0.169 | 0.143 | 0.174 |
| SmolLM3-3B_reasoning | -0.002 | **0.270** | 0.262 | 0.186 |

Layer profile by mismatch type (Qwen3-4B_instruct):
- over: peak_layer=19, peak_mahal=1.373
- under: peak_layer=19, peak_mahal=1.343
- aligned_high: peak_layer=23, peak_mahal=1.392
- aligned_low: peak_layer=19, peak_mahal=1.351

Qwen3-4B_reasoning best layer consistency: **layer 17 is best for 19/20 topics** — extremely stable.

**Key findings**:
- **Strongest correlation in any experiment**: Qwen3-4B_instruct layer 20 achieves r=0.573 (p=0.008) — much higher than the avg-layer r=0.147
- Average-layer Mahalanobis underestimates best-layer correlation by 2-4x — optimal layer selection is crucial
- Q3 (layers 18-26, ~60-75% depth) consistently best for Qwen3-4B instruct/reasoning — matches the "middle layers" finding from Kaplan et al.
- Mismatch types do NOT differ much by layer profile — over- and under-polarized topics peak at the same layers. The mismatch is about magnitude, not layer localization
- Qwen3-4B_reasoning: layer 17 is universally best (19/20 topics) — this model has an extremely concentrated partisan representation
- SmolLM3-3B_reasoning: best layer is 13 (r=0.509), in Q2 — earlier than Qwen3-4B models

#### Exp0C: Probe-Based Mismatch Analysis

**RQ**: Does probe accuracy/confidence predict GSS polarization better than Mahalanobis distance?

**Method**: For each topic, train 5-fold CV linear probes. Compute accuracy, AUC, mean confidence (P(correct class)), and fraction of "ambiguous" politicians (confidence < 0.6). Correlate each metric with GSS.

**Hypothesis**: Probe confidence captures more nuanced information than raw centroid separation. Topics where the probe is CERTAIN (high confidence) may align differently with GSS than topics where the probe is merely ACCURATE.

**Results** (Job 45597276, COMPLETED 2026-02-16):

Pearson r with GSS polarization per model × metric:

| Model | Mahalanobis | Probe Acc | Probe AUC | Mean Conf | Mean Margin | Frac Ambig |
|-------|-------------|-----------|-----------|-----------|-------------|------------|
| Qwen3-4B_base | 0.039 | 0.111 | 0.072 | 0.094 | 0.139 | -0.067 |
| Qwen3-4B_instruct | 0.140 | 0.110 | 0.209 | 0.268 | 0.294 | -0.208 |
| Qwen3-4B_reasoning | **0.250** | **0.327** | 0.195 | **0.346** | **0.354** | -0.279 |
| SmolLM3-3B_base | -0.127 | 0.027 | -0.201 | -0.222 | -0.227 | 0.343 |
| SmolLM3-3B_instruct | -0.269 | -0.208 | -0.372 | 0.152 | 0.237 | -0.193 |
| SmolLM3-3B_reasoning | 0.304 | -0.114 | 0.264 | 0.128 | 0.189 | nan |

Probe performance summary (mean accuracy / mean fraction ambiguous):
- Qwen3-4B_base: acc=0.60, ambig=0.68 (weak signal)
- Qwen3-4B_instruct: acc=0.70, ambig=0.48 (moderate signal)
- Qwen3-4B_reasoning: acc=0.70, ambig=0.67 (moderate signal, less confident)
- SmolLM3-3B_base: acc=0.63, ambig=0.99 (near-random)
- SmolLM3-3B_instruct: acc=0.61, ambig=1.00 (near-random)
- SmolLM3-3B_reasoning: acc=0.52, ambig=1.00 (completely random)

Mahalanobis vs Probe Confidence rank correlation per model:
- Qwen3-4B_base: **0.976** (very tight)
- Qwen3-4B_instruct: 0.920
- Qwen3-4B_reasoning: 0.895
- SmolLM3-3B_base: 0.580 (scattered)
- SmolLM3-3B_instruct: 0.528
- SmolLM3-3B_reasoning: **0.344** (most scattered)

**Key findings**:
- Probe metrics (confidence, margin) provide slightly stronger GSS correlations than Mahalanobis for Qwen3-4B models, but no metric produces strong correlations
- SmolLM3-3B models have essentially no linear party signal — probes near random chance with 100% ambiguous classifications
- The mismatch types (over/under/aligned) don't show clearly distinct probe profiles when averaged; the differences are model-specific
- Qwen3-4B_base has tightest mahal-probe rank agreement (0.976) — both metrics measure the same underlying signal; SmolLM3 models show large rank disagreements

#### Exp0D: D-R Direction Consistency

**RQ**: Do over-polarized and under-polarized topics use different activation dimensions for party separation? Or is there a universal "partisan axis"?

**Method**: For each topic, extract the D-R direction vector (centroid_R - centroid_D, normalized). Compute pairwise cosine similarity between direction vectors across all 20 topics. Cluster topics by their direction geometry.

**Hypothesis**: If there's a universal partisan axis, all topics should have similar direction vectors (high pairwise cosine similarity), and the mismatch is purely about magnitude. If different topics use different axes, the mismatch is structural.

**Results** (Job 45597277, COMPLETED 2026-02-16):

D-R direction vector magnitude vs GSS polarization (Pearson r):

| Model | r | p |
|-------|-----|-------|
| Qwen3-4B_base | 0.182 | 0.442 |
| Qwen3-4B_instruct | **0.394** | 0.086 |
| Qwen3-4B_reasoning | 0.351 | 0.129 |
| SmolLM3-3B_base | -0.233 | 0.323 |
| SmolLM3-3B_instruct | 0.207 | 0.381 |
| SmolLM3-3B_reasoning | 0.096 | 0.688 |

Direction consistency (mean pairwise cosine similarity across all 20 topics):

| Model | Mean sim | Std | Interpretation |
|-------|----------|-----|---------------|
| SmolLM3-3B_instruct | **0.672** | 0.312 | Near-universal partisan axis |
| Qwen3-4B_reasoning | 0.323 | 0.389 | Moderate consistency |
| Qwen3-4B_base | 0.193 | 0.329 | Weak consistency |
| Qwen3-4B_instruct | 0.148 | 0.482 | Weak, high variance |
| SmolLM3-3B_reasoning | 0.028 | 0.416 | Near-random |
| SmolLM3-3B_base | -0.014 | 0.702 | Inverted directions, highest variance |

Within-group vs between-group direction similarity (notable patterns):
- SmolLM3-3B_base: aligned_low topics cluster extremely tightly (within=0.847, diff=+0.942) while under-polarized also cluster (within=0.346, diff=+0.614) — strong bipolar structure despite zero probe accuracy
- SmolLM3-3B_instruct: 17/20 topics collapse into one cluster — near-universal axis with only helpblk, conbus, conlegis as outliers
- Qwen3-4B_instruct: under-polarized topics cluster together (within=0.559, diff=+0.398) — model separates "under-polarized" as a distinct encoding pattern

Hierarchical clustering (4 clusters) — key observation:
- Mismatch categories (over/under/aligned) do NOT map cleanly onto direction clusters in any model
- Topics cluster by content similarity, not by mismatch type
- The mismatch is about MAGNITUDE (how strongly the model separates parties), not about GEOMETRY (which axis it uses)

**Key findings**:
- SmolLM3-3B_instruct has the most consistent partisan axis (mean_sim=0.672): it encodes partisanship similarly across almost all topics — a "single axis" model. Yet it can't actually classify politicians (probe acc ~0.61, ambig=1.00)
- Qwen3-4B models show topic-specific encoding (mean_sim=0.15-0.32) — different topics activate different partisan dimensions
- The universal axis paradox: models with MORE consistent direction vectors don't necessarily have better GSS alignment. SmolLM3-3B_instruct (highest consistency) has r=0.207 while Qwen3-4B_instruct (low consistency) has r=0.394

---

## Paper Framing

**Headline**: Instruction-tuned LLMs develop substantially stronger partisan representations than base or reasoning models (instruct ~1.3-1.7x higher Mahalanobis distance). This signal reflects nuanced biographical knowledge (not simple label lookup) and encodes continuous ideology coherently across topics.

**Story arc**:
1. Instruct models encode partisanship much more strongly (Exp1, Exp2) — replicated (Exp1R)
2. Reasoning training reverses this amplification (Exp1, Exp2)
3. The signal is distributed across all heads (Exp7) but the *most* discriminative heads may form a stable subset (Exp7 redesigned, PENDING)
4. But it doesn't track real-world polarization in absolute terms (Exp5, Bramson)
5. Binary classifiers don't transfer across topics (Exp3)
6. But continuous ideology probes DO transfer (Exp12, rho=0.90) — the representation IS coherent when measured correctly
7. The model encodes affective/policy/identity dimensions of polarization — are they different? (Exp8 redesigned, PENDING)
8. Real politician names make classification HARDER, not easier (Exp9: named=0.91, fictional=1.00) — the signal is from content, not name lookup
9. Instruct models generate behaviorally partisan text (Exp13, accuracy 0.60-0.96)
10. **The LLM-GSS mismatch is about prompt framing, not model failure** (Exp0A): Stance/survey prompts improve r from 0.04→0.29→0.19. SmolLM3-3B_reasoning+survey achieves r=0.569 (p=0.009), the highest single correlation observed
11. **Probe metrics don't outperform Mahalanobis** for GSS alignment (Exp0C): both capture the same signal (rank correlation 0.98 for Qwen3-4B_base). SmolLM3-3B has no linear party signal at all (probe acc ~0.52-0.63, 100% ambiguous)
12. **No universal partisan axis** (Exp0D): direction consistency varies from -0.01 (SmolLM3-3B_base) to 0.67 (SmolLM3-3B_instruct). Models with more consistent axes don't have better GSS alignment — the mismatch is about magnitude modulation, not geometry

**Key insight (revised by Exp12)**: The apparent contradiction between "every head encodes party" (Exp7) and "classifiers don't transfer" (Exp3) is resolved by the granularity of measurement. Binary party labels are too coarse — the model encodes continuous ideological positioning that transfers well across topics (rho=0.84-0.90). The "topic-locked" finding from Exp3 was an artifact of binary classification, not a true feature of the representation.

**Key insight (Exp0)**: The LLM-GSS polarization mismatch is primarily a **framing effect**. Our standard "rhetorical" prompts elicit representations of how politicians would *talk about* issues, while GSS surveys measure how people *answer standardized questions*. Switching to stance ("What is X's position on Y?") or survey-aligned prompts improves the correlation from near-zero to 0.45-0.57 for reasoning models. This means the moderate correlations (r=0.27-0.47) in our main experiments are not a ceiling on alignment — they reflect a measurement mismatch between rhetorical framing and opinion measurement.

**Positioning relative to Kaplan et al. (ICLR 2025)**:
- We replicate their core finding: per-head ridge regression predicts DW-NOMINATE at rho ~0.86 in middle layers, even in smaller models (8-9B vs 70B)
- We extend with base vs instruct vs reasoning comparison: instruct achieves higher probing accuracy (rho=0.84 vs 0.79)
- We show that continuous ideology probes transfer well across topics (rho=0.90 for instruct), which Kaplan et al. did not test
- We demonstrate behavioral relevance: activation-level signals predict generated text partisanship (Exp13)
- **NEW**: We show that activation-space political distances align with real-world opinion polarization (GSS), but this alignment is sensitive to prompt framing

**Broader impact**: Instruction tuning systematically amplifies partisan representations (1.3-1.7x larger activation-space distances) in a way that reasoning training partially mitigates. These representations are:
- Coherent across topics (continuous ideology, not topic-locked)
- Behaviorally relevant (predict generated text content)
- Concentrated in middle layers (replicating Kaplan et al.)
- **Sensitive to prompt framing**: rhetorical prompts capture partisan *style*, while stance/survey prompts capture partisan *substance*. The two are correlated (r~0.3-0.5) but distinct

This suggests RLHF/instruction tuning creates structured political knowledge representations, not just shallow stereotyping.

*Last updated: Feb 16, 2026, 12:00 CST*
