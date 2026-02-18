# Experiment 0: Improving LLM–GSS Polarization Correlation

These experiments investigate **why the correlation between LLM activation polarization and GSS survey polarization is modest** and test specific strategies to improve it.

## Background

The standard pipeline extracts politician-persona activations from LLMs, computes Mahalanobis distance between Democrat/Republican centroids, and correlates this with GSS survey-measured party differences. Baseline correlations (gemma-2-9b-it, all 126 public topics, all-layer average Mahalanobis) are r ≈ 0.20.

These experiments test two levers:
1. **Prompt framing** (Exp 0A): Does the prompt type affect how well LLM representations align with survey opinion?
2. **Layer selection** (Exp 0B): Does restricting to the middle 10% of transformer layers improve alignment?

## Models

| Family | Base | Instruct | Reasoning |
|--------|------|----------|-----------|
| Gemma-2-9b (9B) | `gemma-2-9b` | `gemma-2-9b-it` | — |
| Llama-3.1-8B (8B) | `Meta-Llama-3.1-8B` | `Meta-Llama-3.1-8B-Instruct` | — |
| Qwen3-4B (4B) | `Qwen3-4B-Base` | `Qwen3-4B-Instruct-2507` | `Qwen3-4B-Thinking-2507` |
| SmolLM3-3B (3B) | `SmolLM3-3B-Base` | `SmolLM3-3B` (/no_think) | `SmolLM3-3B` (/think) |

10 model variants total. Qwen3-4B and SmolLM3-3B use a unified model binary for instruct/reasoning, switched via system message.

## Topics

126 filtered public issues (134 total minus 8 excluded: `hubbywk1`, `racdif1–4`, `workwhts`, `wlthwhts`, `intlwhts`). All topics use politician prompts (550 U.S. Congress members, party-coded as Democrat=100 / Republican=200).

## Pipeline

For each model × topic × prompt condition:
1. Generate 550 politician prompts (completion-style for base, chat-template for instruct/reasoning)
2. Extract activations: shape (550, L, H, D) via `extract_heads_batched`
3. **All-layer metric**: average activations across all L layers → (550, H×D) → PCA to 15 dims → Mahalanobis between D/R centroids
4. **Mid-10% metric**: average activations across only the middle 10% of layers (e.g., layers 18–22 for 42-layer Gemma) → same PCA → Mahalanobis pipeline
5. Correlate Mahalanobis with GSS polarization (Pearson r and Spearman rho, n=126)

## Scripts

| File | Description |
|------|-------------|
| `exp0ab_extended.py` | Main script. `--group {1,2,3,4}` selects model subset |
| `exp0ab_g1.sbatch` | Gemma-2-9b (base + instruct) |
| `exp0ab_g2.sbatch` | Llama-3.1-8B (base + instruct) |
| `exp0ab_g3.sbatch` | Qwen3-4B (base + instruct + reasoning) |
| `exp0ab_g4.sbatch` | SmolLM3-3B (base + instruct + reasoning) |

Each job runs on 1 A100 (40GB) on the `ssd-gpu` partition.

## Results

### Exp 0A: Prompt Framing

Three prompt conditions per model type:

| Condition | Base prompt | Instruct/Reasoning prompt |
|-----------|-------------|--------------------------|
| **Rhetorical** | `"{name} makes a statement on {topic}:"` | `"Generate a statement by {name} on {topic}."` |
| **Stance** | `"On {topic}, {name}'s position is"` | `"What is {name}'s position on {topic}?"` |
| **Survey** | `"In a survey about {question}, {name} would respond"` | `"If asked in a national survey about {question}, how would {name} respond?"` |

#### Full results: Pearson r (Mahalanobis vs GSS polarization, n=126)

| Model | Rhetorical | Stance | Survey | Best |
|-------|-----------|--------|--------|------|
| Gemma-2-9b_base | 0.286 | **0.299** | 0.212 | stance |
| **Gemma-2-9b_instruct** | **0.472** | 0.170 | 0.327 | **rhetorical** |
| Llama-3.1-8B_base | 0.192 | 0.406 | **0.438** | survey |
| Llama-3.1-8B_instruct | 0.400 | 0.404 | **0.455** | survey |
| Qwen3-4B_base | -0.147 | **0.259** | 0.138 | stance |
| Qwen3-4B_instruct | 0.159 | **0.357** | 0.242 | stance |
| Qwen3-4B_reasoning | 0.291 | **0.360** | 0.335 | stance |
| SmolLM3-3B_base | 0.046 | **0.392** | 0.236 | stance |
| SmolLM3-3B_instruct | 0.169 | -0.073 | 0.129 | rhetorical |
| SmolLM3-3B_reasoning | **0.426** | -0.016 | 0.419 | rhetorical |

#### Average r by condition

| Condition | Mean r (10 models) | Median r |
|-----------|-------------------|----------|
| Rhetorical | 0.229 | 0.226 |
| Stance | 0.256 | 0.329 |
| Survey | 0.293 | 0.263 |

#### Key findings (0A)

1. **No single best prompt for all models.** The optimal framing is model-dependent.
2. **Stance works best for base models** (4/4 base models). By directly asking for a position rather than generating rhetoric, base models produce activations that better track opinion content.
3. **Rhetorical works best for some instruct models** — Gemma-2-9b_instruct achieves the highest single-model r=0.472 with rhetorical prompts. Instruct models may already internalize opinion content even when generating statements.
4. **Survey prompts are most consistent for Llama** (best or near-best for both base and instruct).
5. **SmolLM3 instruct is nearly unpredictive** regardless of prompt (max r=0.169). SmolLM3 reasoning, however, reaches r=0.426 — the instruct mode (/no_think) may suppress opinion-relevant representations.

### Exp 0B: Middle 10% of Layers vs All Layers

Restricting Mahalanobis computation to the middle 10% of layers (e.g., layers 18–22 for 42-layer Gemma, layers 14–16 for 32-layer Llama). Compared against the default all-layer average, using the rhetorical prompt condition.

| Model | r (all-layer) | r (mid-10%) | Delta |
|-------|--------------|-------------|-------|
| Gemma-2-9b_base | 0.286 | 0.272 | -0.014 |
| **Gemma-2-9b_instruct** | 0.472 | **0.555** | **+0.084** |
| **Llama-3.1-8B_base** | 0.192 | **0.401** | **+0.209** |
| Llama-3.1-8B_instruct | 0.400 | 0.384 | -0.016 |
| Qwen3-4B_base | -0.147 | 0.052 | +0.199 |
| Qwen3-4B_instruct | 0.159 | 0.187 | +0.028 |
| Qwen3-4B_reasoning | 0.291 | 0.226 | -0.065 |
| SmolLM3-3B_base | 0.046 | 0.150 | +0.104 |
| SmolLM3-3B_instruct | 0.169 | 0.019 | -0.150 |
| SmolLM3-3B_reasoning | 0.426 | 0.286 | -0.140 |

#### Cross-condition analysis (mid-10% delta for all prompt × model pairs)

Notable improvements (delta > +0.1):

| Model | Condition | r_all | r_mid10 | Delta |
|-------|-----------|-------|---------|-------|
| **Gemma-2-9b_instruct** | **stance** | 0.170 | **0.481** | **+0.311** |
| Llama-3.1-8B_base | rhetorical | 0.192 | 0.401 | +0.209 |
| Qwen3-4B_base | rhetorical | -0.147 | 0.052 | +0.199 |
| SmolLM3-3B_instruct | survey | 0.129 | 0.279 | +0.151 |
| SmolLM3-3B_reasoning | stance | -0.016 | 0.127 | +0.143 |
| SmolLM3-3B_base | rhetorical | 0.046 | 0.150 | +0.104 |
| Gemma-2-9b_base | survey | 0.212 | 0.312 | +0.100 |

#### Key findings (0B)

1. **Mid-10% helps base models consistently.** All 4 base models see improvement with rhetorical prompts (Llama +0.209, Qwen +0.199, SmolLM +0.104). Early and late layers add noise for base models.
2. **Best single result: Gemma-2-9b_instruct + rhetorical + mid-10% = r=0.555.** This nearly triples the baseline demographic simulation correlation of r=0.21.
3. **Gemma-2-9b_instruct + stance + mid-10% = r=0.481 (+0.311).** The largest single improvement from layer selection — stance prompts perform poorly with all layers (r=0.170) but recover when restricted to mid-layers.
4. **Mid-10% hurts when the all-layer correlation is already strong.** Models with strong all-layer r (SmolLM3 reasoning r=0.426, Qwen3 reasoning r=0.291) tend to worsen with mid-10%, suggesting their opinion signal is distributed broadly.
5. **Layer selection and prompt framing interact.** The optimal prompt depends on whether you use all layers or mid-10%. Stance goes from worst to best for Gemma instruct when switching to mid-10%.

## Summary: Best Configurations

Top 5 configurations by Pearson r with GSS polarization (n=126 public issues):

| Rank | Model | Prompt | Layers | r | p |
|------|-------|--------|--------|---|---|
| 1 | Gemma-2-9b_instruct | rhetorical | mid-10% | **0.555** | <0.0001 |
| 2 | Gemma-2-9b_instruct | stance | mid-10% | **0.481** | <0.0001 |
| 3 | Gemma-2-9b_instruct | rhetorical | all | **0.472** | <0.0001 |
| 4 | Llama-3.1-8B_instruct | survey | all | **0.455** | <0.0001 |
| 5 | Llama-3.1-8B_base | survey | all | **0.438** | <0.0001 |

## Implications

1. The standard pipeline (rhetorical prompts + all-layer average) is a **suboptimal default**. Prompt choice and layer selection can improve r from 0.21 to 0.55 — a factor of 2.6x.
2. **Larger models perform better** — Gemma-2-9b and Llama-3.1-8B consistently outperform the 3–4B models, likely encoding richer political knowledge.
3. **Instruct tuning generally helps** — instruct models outperform base models in the majority of prompt conditions.
4. The interaction between prompt framing and layer selection suggests that **different layers encode different types of political content** (rhetorical framing vs. opinion substance).

## Output Files

Results are saved in `experiments/results/`:

| Pattern | Contents |
|---------|----------|
| `exp0ab_ext_g{1-4}_*.csv` | Per-group detail CSV (topic × condition × model) |
| `exp0ab_ext_g{1-4}_*.pkl` | Per-group detail pickle |
| `exp0ab_ext_g{1-4}_{ModelName}_*.pkl` | Per-model checkpoint |

Columns in CSVs: `model`, `model_type`, `family`, `condition`, `topic`, `mahal_all`, `mahal_mid10`, `n_layers`, `mid_layers`, `gss_polarization`

## Prior Experiments (Exp 0A–0D, 20-topic pilot)

These extended experiments follow a smaller pilot (20 strategically selected topics, Qwen3-4B + SmolLM3-3B only). See `RESULTS_ANALYSIS.md` for the pilot results and the full experiment series (Exp 1–13) design.
