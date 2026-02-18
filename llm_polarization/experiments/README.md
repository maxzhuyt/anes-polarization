# Experiment 0A/0B: Improving LLM–GSS Polarization Correlation

These experiments test whether **prompt framing** and **layer selection** can improve the correlation between LLM activation polarization (Mahalanobis distance) and GSS survey polarization. We run both **politician simulation** and **demographic simulation** variants.

## Background

The standard pipeline (`run_model_comparison.py`) extracts politician-persona activations from LLMs, computes per-head Mahalanobis distance between Democrat/Republican centroids across all L×H attention heads, and correlates the average Mahalanobis with GSS survey-measured party differences.

Baseline correlations (per-head Mahalanobis, PCA-15, mean centroid, 126 public topics):

| Model | Baseline r |
|-------|-----------|
| Gemma-2-9b_instruct | 0.648 |
| Llama-3.1-8B_instruct | 0.578 |
| Qwen3-4B_reasoning | 0.470 |
| SmolLM3-3B_base | 0.458 |

These experiments test two levers:
1. **Prompt framing** (Exp 0A): Does the prompt type (rhetorical/stance/survey) affect how well LLM representations align with survey opinion?
2. **Layer selection** (Exp 0B): Does restricting to the middle 10% of layers improve alignment?

## Models

| Family | Base | Instruct | Reasoning |
|--------|------|----------|-----------|
| Gemma-2-9b (9B) | `gemma-2-9b` | `gemma-2-9b-it` | — |
| Llama-3.1-8B (8B) | `Meta-Llama-3.1-8B` | `Meta-Llama-3.1-8B-Instruct` | — |
| Qwen3-4B (4B) | `Qwen3-4B-Base` | `Qwen3-4B-Instruct-2507` | `Qwen3-4B-Thinking-2507` |
| SmolLM3-3B (3B) | `SmolLM3-3B-Base` | `SmolLM3-3B` (/no_think) | `SmolLM3-3B` (/think) |

10 model variants total. Qwen3-4B and SmolLM3-3B use a unified model binary for instruct/reasoning, switched via system message.

## Topics

126 filtered public issues (134 total minus 8 excluded: `hubbywk1`, `racdif1–4`, `workwhts`, `wlthwhts`, `intlwhts`).

## Pipeline

### Politician simulation (`exp0ab_extended.py`)

For each model × topic × prompt condition:
1. Generate 550 politician prompts using `fullname` (e.g., "Donald Trump"), completion-style for base models, chat-template for instruct/reasoning
2. Extract activations: shape (550, L, H, D) via `extract_heads_batched`, max_length=128
3. Compute **per-head Mahalanobis grid** (L, H) via `compute_all_head_metrics_pca` — each head (N, D=128) is independently PCA-reduced to 15 dims, then Mahalanobis between D/R centroids is computed
4. **All-layer metric**: `np.mean(grid)` — average of all L×H head distances
5. **Mid-10% metric**: `np.mean(grid[mid_start:mid_end, :])` — average of middle 10% of layers only
6. Correlate with GSS polarization (Pearson r and Spearman rho, n=126)

### Demographic simulation (`exp0ab_demo.py`)

For each model × topic:
1. Load GSS respondents, stratified sample at 10% (~1500 per topic)
2. Build demographic profiles from 83 fields (prompt format B)
3. Extract activations: shape (~1500, L, H, D), max_length=512
4. Same per-head Mahalanobis grid → all-layer and mid-10% metrics (both mean and median centroids)
5. Correlate with GSS polarization

## Scripts

| File | Description |
|------|-------------|
| `exp0ab_extended.py` | Politician simulation. `--group {1,2,3,4}` selects model subset |
| `exp0ab_demo.py` | Demographic simulation. `--group {1,2,3,4}` selects model subset |
| `exp0ab_g{1-4}.sbatch` | Politician sbatch files (ssd-gpu, A100, 6h, 8G) |
| `exp0ab_demo_g{1-4}.sbatch` | Demographic sbatch files (ssd-gpu, A100, 18h, 16G) |

Each group runs on 1 A100 (40GB) on the `ssd-gpu` partition:
- Group 1: Gemma-2-9b (base + instruct)
- Group 2: Llama-3.1-8B (base + instruct)
- Group 3: Qwen3-4B (base + instruct + reasoning)
- Group 4: SmolLM3-3B (base + instruct + reasoning)

## Results: Politician Simulation

### Exp 0A: Prompt Framing

Three prompt conditions per model type:

| Condition | Base prompt | Instruct/Reasoning prompt |
|-----------|-------------|--------------------------|
| **Rhetorical** | `"{name} makes a statement on {topic}:"` | `"Generate a statement by {name} on {topic}."` |
| **Stance** | `"On {topic}, {name}'s position is"` | `"What is {name}'s position on {topic}?"` |
| **Survey** | `"In a survey about {question}, {name} would respond"` | `"If asked in a national survey about {question}, how would {name} respond?"` |

#### Full results: Pearson r (all-layer per-head Mahalanobis vs GSS polarization, n=126)

| Model | Rhetorical | Stance | Survey | Best |
|-------|-----------|--------|--------|------|
| Gemma-2-9b_base | **0.552** | 0.499 | 0.536 | rhetorical |
| **Gemma-2-9b_instruct** | **0.647** | 0.572 | 0.557 | **rhetorical** |
| Llama-3.1-8B_base | 0.456 | **0.566** | 0.544 | stance |
| **Llama-3.1-8B_instruct** | 0.577 | 0.607 | **0.645** | **survey** |
| Qwen3-4B_base | 0.271 | **0.584** | 0.390 | stance |
| Qwen3-4B_instruct | 0.334 | **0.579** | 0.295 | stance |
| **Qwen3-4B_reasoning** | 0.468 | **0.605** | 0.565 | **stance** |
| SmolLM3-3B_base | 0.456 | **0.496** | 0.484 | stance |
| SmolLM3-3B_instruct | 0.395 | 0.354 | **0.555** | survey |
| **SmolLM3-3B_reasoning** | 0.399 | 0.379 | **0.613** | **survey** |

#### Key findings (0A)

1. **Prompt choice substantially affects correlation.** The best prompt per model ranges from r=0.496 to r=0.647, while the worst can be as low as r=0.271.
2. **Rhetorical is best for Gemma** (both base and instruct). Gemma-2-9b_instruct + rhetorical = 0.647 matches the baseline.
3. **Stance is best for Qwen and Llama base.** Qwen3-4B_base jumps from 0.271 (rhetorical) to 0.584 (stance) — a +0.313 improvement.
4. **Survey is best for SmolLM3 instruct/reasoning and Llama instruct.** SmolLM3-3B_reasoning + survey = 0.613 is a +0.214 improvement over its baseline (rhetorical = 0.399).
5. **No single best prompt across all models.** The optimal framing is model-family dependent.

### Exp 0B: Middle 10% of Layers vs All Layers

Mid-10% layer selection applied to all prompt × model combinations. Showing the rhetorical condition for direct baseline comparison:

| Model | r (all-layer) | r (mid-10%) | Delta |
|-------|--------------|-------------|-------|
| Gemma-2-9b_base | 0.552 | 0.432 | -0.119 |
| Gemma-2-9b_instruct | 0.647 | 0.634 | -0.013 |
| Llama-3.1-8B_base | 0.456 | 0.425 | -0.031 |
| Llama-3.1-8B_instruct | 0.577 | 0.564 | -0.013 |
| Qwen3-4B_base | 0.271 | 0.252 | -0.019 |
| Qwen3-4B_instruct | 0.334 | 0.287 | -0.047 |
| Qwen3-4B_reasoning | 0.468 | 0.278 | -0.190 |
| SmolLM3-3B_base | 0.456 | 0.468 | +0.012 |
| SmolLM3-3B_instruct | 0.395 | 0.412 | +0.017 |
| SmolLM3-3B_reasoning | 0.399 | 0.410 | +0.011 |

#### Notable mid-10% improvements across all conditions

| Model | Condition | r_all | r_mid10 | Delta |
|-------|-----------|-------|---------|-------|
| Gemma-2-9b_instruct | stance | 0.572 | **0.628** | **+0.057** |
| SmolLM3-3B_reasoning | stance | 0.379 | **0.427** | **+0.048** |
| Llama-3.1-8B_instruct | rhetorical | 0.577 | 0.564 | -0.013 |
| Llama-3.1-8B_instruct | survey | 0.645 | 0.641 | -0.004 |

#### Key findings (0B)

1. **Mid-10% generally does not help with the per-head approach.** Unlike the earlier (incorrect) layer-average concatenation approach, restricting to mid-layers provides minimal benefit when using per-head Mahalanobis.
2. **SmolLM3 is the exception** — all three variants show small mid-10% improvements with rhetorical prompts, possibly because the 30-layer architecture has a cleaner mid-layer signal.
3. **Mid-10% actively hurts Qwen3-4B_reasoning** (delta = -0.190 to -0.211), suggesting its opinion signal is distributed broadly across layers.
4. **All-layer is the safer default.** In 22 of 30 model×condition pairs, all-layer matches or beats mid-10%.

## Results: Demographic Simulation

Demographic simulation correlations are substantially lower than politician simulation across all models:

| Model | all-mean | mid10-mean | all-median | mid10-median | Best r |
|-------|---------|-----------|-----------|-------------|--------|
| Gemma-2-9b_base | 0.076 | **0.182** | 0.040 | 0.115 | 0.182 |
| Gemma-2-9b_instruct | 0.197 | 0.134 | **0.211** | 0.158 | 0.211 |
| Llama-3.1-8B_base | 0.025 | 0.042 | 0.060 | **0.078** | 0.078 |
| Llama-3.1-8B_instruct | 0.199 | **0.216** | 0.132 | 0.155 | 0.216 |
| Qwen3-4B_base | -0.068 | -0.017 | 0.034 | **0.050** | 0.050 |
| Qwen3-4B_instruct | **0.184** | 0.070 | 0.134 | 0.072 | 0.184 |
| Qwen3-4B_reasoning | -0.100 | -0.095 | 0.005 | **0.017** | 0.017 |
| SmolLM3-3B_base | 0.006 | **0.107** | 0.053 | 0.086 | 0.107 |
| SmolLM3-3B_instruct | **0.162** | 0.117 | 0.098 | 0.114 | 0.162 |
| SmolLM3-3B_reasoning | 0.022 | 0.062 | 0.074 | **0.104** | 0.104 |

Best demographic result: **Llama-3.1-8B_instruct = r=0.216** (mid10-mean).

## Summary: Politician vs Demographic

| Model | Politician (best r) | Best prompt | Demographic (best r) | Ratio |
|-------|-------------------|-------------|---------------------|-------|
| Gemma-2-9b_base | **0.552** | rhetorical | 0.182 | 3.0x |
| Gemma-2-9b_instruct | **0.647** | rhetorical | 0.211 | 3.1x |
| Llama-3.1-8B_base | **0.566** | stance | 0.078 | 7.3x |
| Llama-3.1-8B_instruct | **0.645** | survey | 0.216 | 3.0x |
| Qwen3-4B_base | **0.584** | stance | 0.050 | 11.7x |
| Qwen3-4B_instruct | **0.579** | stance | 0.184 | 3.1x |
| Qwen3-4B_reasoning | **0.605** | stance | 0.017 | 35.6x |
| SmolLM3-3B_base | **0.496** | stance | 0.107 | 4.6x |
| SmolLM3-3B_instruct | **0.555** | survey | 0.162 | 3.4x |
| SmolLM3-3B_reasoning | **0.613** | survey | 0.104 | 5.9x |

## Methodological note: Name format matters

An early run accidentally used `bioname` (e.g., "TRUMP, Donald John") instead of `fullname` (e.g., "Donald Trump"). This provides a natural ablation of name recognizability:

| Model | fullname r | bioname r | Delta |
|-------|-----------|----------|-------|
| Gemma-2-9b_base | 0.552 | 0.322 | **-0.230** |
| Gemma-2-9b_instruct | 0.647 | 0.555 | -0.092 |
| Llama-3.1-8B_base | 0.456 | 0.394 | -0.062 |
| Llama-3.1-8B_instruct | 0.577 | 0.551 | -0.026 |
| Qwen3-4B_base | 0.271 | 0.046 | **-0.225** |
| Qwen3-4B_instruct | 0.334 | 0.245 | -0.089 |
| Qwen3-4B_reasoning | 0.468 | 0.317 | **-0.151** |
| SmolLM3-3B_base | 0.456 | 0.269 | **-0.187** |
| SmolLM3-3B_instruct | 0.395 | 0.383 | -0.012 |
| SmolLM3-3B_reasoning | 0.399 | 0.469 | +0.070 |
| **Mean** | | | **-0.100** |

Base models are far more sensitive to name format (avg -0.176) than instruct models (avg -0.022). The ALL-CAPS citation format likely prevents LLMs from activating politician-specific knowledge stored under natural name tokens during pretraining.

## Conclusions

1. **Prompt framing provides genuine improvement.** Choosing the right prompt per model family can improve r by +0.1 to +0.3 over the default rhetorical prompt. Stance works best for Qwen, survey for SmolLM3/Llama instruct, rhetorical for Gemma.
2. **Mid-10% layer selection does not help with the per-head approach.** When each head is independently PCA-reduced, restricting to mid-layers discards useful signal rather than noise.
3. **Politician simulation far outperforms demographic simulation** (3–36x higher r). LLMs encode strong partisan knowledge under politician names; demographic profiles provide much weaker ideological signal.
4. **Larger models perform better.** Gemma-2-9b and Llama-3.1-8B consistently outperform the 3–4B models.
5. **Instruct tuning generally helps** but is not universal — SmolLM3-3B_base (r=0.496) outperforms SmolLM3-3B_instruct (r=0.395) with stance prompts.
6. **Name format is a significant confound.** Using formal citation names vs natural names can degrade r by up to 0.23, especially for base models.

## Output Files

Results are saved in `results/`:

| Pattern | Contents |
|---------|----------|
| `exp0ab_ext_g{1-4}_*.csv` | Politician detail CSV (topic × condition × model) |
| `exp0ab_demo_g{1-4}_*.csv` | Demographic detail CSV (topic × model) |
| `*_{ModelName}_*.pkl` | Per-model checkpoint pickles |

Politician CSV columns: `model`, `model_type`, `family`, `condition`, `topic`, `mahal_all`, `mahal_mid10`, `n_layers`, `mid_layers`, `gss_polarization`

Demographic CSV columns: `model`, `model_type`, `family`, `topic`, `n_sampled`, `n_dem`, `n_rep`, `n_layers`, `mid_layers`, `mahal_all`, `mahal_mid10`, `mahal_max`, `mahal_all_median`, `mahal_mid10_median`, `gss_polarization`

## Prior Experiments (Exp 0A–0D, 20-topic pilot)

These extended experiments follow a smaller pilot (20 strategically selected topics, Qwen3-4B + SmolLM3-3B only). See `RESULTS_ANALYSIS.md` for the pilot results and the full experiment series (Exp 1–13) design.
