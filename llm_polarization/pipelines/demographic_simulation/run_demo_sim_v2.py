"""
GSS Demographic Simulation V2: Prompt Framing + Layer Selection

Same 10 model variants as run_gss_pca_v2.py (politician), but uses GSS respondent
demographic profiles instead of politician names. This tests whether LLM representations
of *anonymous demographic personas* align with survey-measured political polarization.

Configuration (best from test_config runs):
  - demo_fields: all (83 fields)
  - prompt_fmt: B ("Given the following background about a person...")
  - sample_frac: 0.1 (~1500 respondents/topic)
  - max_length: 512
  - system_msg: "You are simulating the views of an American."

Computes per-head Mahalanobis grid (L, H), then:
  - all-layer: mean across entire grid
  - mid-10%: mean across middle 10% of layers only

Usage:
    python run_demo_sim_v2.py --group 1   # gemma-2-9b (base + instruct)
    python run_demo_sim_v2.py --group 2   # Llama-3.1-8B (base + instruct)
    python run_demo_sim_v2.py --group 3   # Qwen3-4B (base + instruct + reasoning)
    python run_demo_sim_v2.py --group 4   # SmolLM3-3B (base + instruct + reasoning)
"""

import os
import sys
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gc
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from model_utils import load_model, extract_heads_batched
from run_gss_pca import compute_all_head_metrics_pca

# =============================================================================
# Paths
# =============================================================================

DATA_ROOT = Path("/project/jevans/maxzhuyt/gss_polarization")
MODELS_DIR = "/project/jevans/maxzhuyt/models"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GSS_CSV = DATA_ROOT / "data" / "gss" / "gss_2021_2024.csv"
STATA_2024 = DATA_ROOT / "data" / "gss" / "GSS2024.dta"
STATA_2022 = DATA_ROOT / "data" / "gss" / "GSS2022.dta"
DEMO_CSV = DATA_ROOT / "question_lists" / "gss_demographic_variables.csv"
PUBLIC_TOPICS_CSV = DATA_ROOT / "question_lists" / "public_issues.csv"
PUBLIC_POL_CSV = DATA_ROOT / "data" / "polarization" / "public_issues_polarization.csv"

EXPERIMENT_NAME = "demo_sim_v2"
SYSTEM_MSG_DEMO = "You are simulating the views of an American."
PROMPT_FMT = "Given the following background about a person:\n{profile}\n\nHow would they answer: {question}"
SAMPLE_FRAC = 0.1
SAMPLE_SEED = 42
MAX_LENGTH = 512
PCA_DIM = 15

EXCLUDED_PUBLIC = [
    "hubbywk1", "racdif1", "racdif2", "racdif3", "racdif4",
    "workwhts", "wlthwhts", "intlwhts",
]

# All 83 demographic variables present in all GSS years
DEMO_VARS_ALL_YEARS = {
    'age', 'agekdbrn', 'babies', 'born', 'childs', 'degree', 'denom', 'denom16',
    'dipged', 'divorce', 'earnrs', 'educ', 'evwork', 'famdif16', 'family16',
    'granborn', 'health', 'hompop', 'hrs1', 'hrs2', 'incom16', 'income',
    'indus10', 'madeg', 'maeduc', 'maind10', 'major1', 'maocc10', 'marital',
    'mawrkgrw', 'mawrkslf', 'mobile16', 'numemps', 'occ10', 'othlang',
    'othlang1', 'othlang2', 'padeg', 'paeduc', 'paind10', 'paocc10', 'parborn',
    'partfull', 'pawrkslf', 'polviews', 'posslq', 'posslqy', 'preteen', 'race',
    'reg16', 'region', 'relig', 'relig16', 'res16', 'rincome', 'sex', 'sexornt',
    'sibs', 'spdeg', 'spden', 'speduc', 'spevwork', 'sphrs1', 'sphrs2',
    'spind10', 'spklang', 'spocc10', 'sprel', 'spwrkslf', 'spwrksta', 'teens',
    'unemp', 'unrelat', 'weekswrk', 'widowed', 'wksub', 'wksubs', 'wksup',
    'wksups', 'wrkslf', 'wrkstat', 'xnorcsiz', 'yousup',
}

STRATIFY_COLS = ['polviews', 'age_bin', 'degree', 'race', 'sex', 'rincome']

# Batch sizes for A100 40GB at max_length=512
MODEL_GROUPS = {
    1: {
        "Gemma-2-9b_base": {
            "path": f"{MODELS_DIR}/gemma-2-9b",
            "type": "base", "batch_size": 12, "family": "Gemma-2-9b",
        },
        "Gemma-2-9b_instruct": {
            "path": f"{MODELS_DIR}/gemma-2-9b-it",
            "type": "instruct", "batch_size": 12, "family": "Gemma-2-9b",
        },
    },
    2: {
        "Llama-3.1-8B_base": {
            "path": f"{MODELS_DIR}/Meta-Llama-3.1-8B",
            "type": "base", "batch_size": 40, "family": "Llama-3.1-8B",
        },
        "Llama-3.1-8B_instruct": {
            "path": f"{MODELS_DIR}/Meta-Llama-3.1-8B-Instruct",
            "type": "instruct", "batch_size": 40, "family": "Llama-3.1-8B",
        },
    },
    3: {
        "Qwen3-4B_base": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Base",
            "type": "base", "batch_size": 64, "family": "Qwen3-4B",
        },
        "Qwen3-4B_instruct": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Instruct-2507",
            "type": "instruct", "batch_size": 64, "family": "Qwen3-4B",
        },
        "Qwen3-4B_reasoning": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Thinking-2507",
            "type": "reasoning", "batch_size": 64, "family": "Qwen3-4B",
        },
    },
    4: {
        "SmolLM3-3B_base": {
            "path": f"{MODELS_DIR}/SmolLM3-3B-Base",
            "type": "base", "batch_size": 80, "family": "SmolLM3-3B",
        },
        "SmolLM3-3B_instruct": {
            "path": f"{MODELS_DIR}/SmolLM3-3B",
            "type": "instruct", "batch_size": 80, "family": "SmolLM3-3B",
            "system_override": "/no_think",
        },
        "SmolLM3-3B_reasoning": {
            "path": f"{MODELS_DIR}/SmolLM3-3B",
            "type": "reasoning", "batch_size": 80, "family": "SmolLM3-3B",
            "system_override": "/think",
        },
    },
}


# =============================================================================
# GSS data loading (adapted from test_config.py)
# =============================================================================

def build_code_maps(all_fields):
    """Build code->label mappings from Stata files."""
    print("Loading value labels from Stata files...")
    df_num_24 = pd.read_stata(str(STATA_2024), convert_categoricals=False)
    df_cat_24 = pd.read_stata(str(STATA_2024), convert_categoricals=True)
    df_num_22 = pd.read_stata(str(STATA_2022), convert_categoricals=False)
    reader_22 = pd.io.stata.StataReader(str(STATA_2022))
    vl_22 = reader_22.value_labels()
    var_to_lbl_22 = dict(zip(reader_22._varlist, reader_22._lbllist))

    def _build_one(var_name, max_code=100000):
        if var_name in df_num_24.columns:
            mapping = {}
            for n, c in zip(df_num_24[var_name], df_cat_24[var_name]):
                if pd.notna(n) and pd.notna(c) and int(n) < max_code:
                    mapping[int(n)] = str(c).strip()
            if mapping:
                return mapping
        if var_name in df_num_22.columns:
            lbl = var_to_lbl_22.get(var_name, '')
            if lbl and lbl in vl_22:
                return {int(k): str(v).strip() for k, v in vl_22[lbl].items() if int(k) < max_code}
        return {}

    code_maps = {field: _build_one(field) for field in all_fields}
    del df_num_24, df_cat_24, df_num_22
    gc.collect()
    return code_maps


def precompute_demo_parts(df, code_maps, all_fields, all_labels):
    """Pre-compute (field_name, 'Label: value') tuples per respondent."""
    all_parts = {}
    for idx in df.index:
        row = df.loc[idx]
        parts = []
        for field in all_fields:
            val = row.get(field)
            if pd.isna(val):
                continue
            code = int(val)
            label = all_labels.get(field, field)
            if field in code_maps and code in code_maps[field]:
                text = code_maps[field][code]
            else:
                text = str(code)
            parts.append((field, f'{label}: {text}'))
        all_parts[idx] = parts
    return all_parts


def stratified_sample(df, frac, seed, min_per_group=1):
    """Stratified sample within each _strat_key group."""
    rng = np.random.default_rng(seed)
    sampled = []
    for _, group in df.groupby('_strat_key'):
        n = max(min_per_group, int(np.ceil(len(group) * frac)))
        n = min(n, len(group))
        idx = rng.choice(group.index, size=n, replace=False)
        sampled.append(df.loc[idx])
    return pd.concat(sampled).sort_index()


def is_valid_response(val):
    if pd.isna(val):
        return False
    try:
        return int(float(val)) < 1000000
    except (ValueError, TypeError):
        return False


def load_gss_data(active_fields, all_labels):
    """Load GSS data, filter D/R, build stratification, precompute profiles."""
    print("Loading GSS data...")
    df_gss = pd.read_csv(str(GSS_CSV), low_memory=False)
    print(f"  Respondents: {df_gss.shape[0]}, Variables: {df_gss.shape[1]}")

    DEM_CODES = [0, 1, 2]
    REP_CODES = [4, 5, 6]
    df_gss['party_code'] = np.nan
    df_gss.loc[df_gss['partyid'].isin(DEM_CODES), 'party_code'] = 100
    df_gss.loc[df_gss['partyid'].isin(REP_CODES), 'party_code'] = 200
    df_dr = df_gss[df_gss['party_code'].notna()].copy()
    n_dem = int((df_dr['party_code'] == 100).sum())
    n_rep = int((df_dr['party_code'] == 200).sum())
    print(f"  D/R respondents: {len(df_dr)} (D={n_dem}, R={n_rep})")

    # Stratification
    df_dr['age_bin'] = pd.cut(df_dr['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=False)
    strat_cols = [c for c in STRATIFY_COLS if c in df_dr.columns]
    df_dr['_strat_key'] = ''
    for col in strat_cols:
        df_dr['_strat_key'] += df_dr[col].fillna(-1).astype(int).astype(str) + '_'
    print(f"  Stratification groups: {df_dr['_strat_key'].nunique()}")

    # Code maps + demographic profiles
    code_maps = build_code_maps(active_fields)
    print("Pre-computing demographic profiles...")
    demo_parts = precompute_demo_parts(df_dr, code_maps, active_fields, all_labels)
    print(f"  Done: {len(demo_parts)} respondents")

    return df_dr, demo_parts


def load_filtered_topics():
    """Load 126 filtered public topics that have polarization data."""
    pub_topics = pd.read_csv(str(PUBLIC_TOPICS_CSV))
    pol_pub = pd.read_csv(str(PUBLIC_POL_CSV))

    pol_valid = set(pol_pub['variable'])
    topics = {}
    for _, row in pub_topics.iterrows():
        var = row['Variable']
        if var in pol_valid and var.lower() not in [e.lower() for e in EXCLUDED_PUBLIC]:
            topics[var] = row['SurveyQuestion']
    return topics, pol_pub


# =============================================================================
# Per-topic analysis
# =============================================================================

def run_topic(model, tokenizer, topic_name, survey_question,
              df_dr, demo_parts, active_fields_set,
              system_msg, batch_size):
    """Run a single topic: extract activations, compute all-layer and mid-10% metrics."""
    t0 = time.time()

    if topic_name not in df_dr.columns:
        return None

    valid_mask = df_dr[topic_name].apply(is_valid_response)
    df_valid = df_dr[valid_mask]
    if len(df_valid) < 20:
        return None

    df_sampled = stratified_sample(df_valid, SAMPLE_FRAC, SAMPLE_SEED)
    n_dem = int((df_sampled['party_code'] == 100).sum())
    n_rep = int((df_sampled['party_code'] == 200).sum())
    if n_dem < 5 or n_rep < 5:
        return None

    # Exclude topic from persona if it's a demographic field
    exclude_field = topic_name if topic_name in active_fields_set else None

    # Build prompts
    rng = np.random.default_rng(hash(topic_name) % (2**32))
    prompts = []
    for idx in df_sampled.index:
        if exclude_field:
            parts = [p[1] for p in demo_parts[idx] if p[0] != exclude_field]
        else:
            parts = [p[1] for p in demo_parts[idx]]
        rng.shuffle(parts)
        profile = '. '.join(parts) + '.'
        prompts.append(PROMPT_FMT.format(profile=profile, question=survey_question))

    labels = df_sampled['party_code'].values.astype(int)

    # Extract activations
    X_heads = extract_heads_batched(
        model, tokenizer, prompts, system_msg,
        batch_size=batch_size, max_length=MAX_LENGTH,
    )

    N, L, H, D = X_heads.shape

    # Per-head Mahalanobis grid (same as test_config baseline)
    grid_mean = compute_all_head_metrics_pca(
        X_heads, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='mean',
    )  # (L, H)
    grid_median = compute_all_head_metrics_pca(
        X_heads, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='median',
    )

    # All-layer and mid-10% metrics
    mid_start = int(L * 0.45)
    mid_end = max(int(L * 0.55), mid_start + 1)

    result = {
        'topic': topic_name,
        'n_sampled': len(df_sampled),
        'n_dem': n_dem,
        'n_rep': n_rep,
        'n_layers': L,
        'mid_layers': f"{mid_start}-{mid_end-1}",
        # Mean centroid
        'mahal_all': float(np.mean(grid_mean)),
        'mahal_mid10': float(np.mean(grid_mean[mid_start:mid_end, :])),
        'mahal_max': float(np.max(grid_mean)),
        # Median centroid
        'mahal_all_median': float(np.mean(grid_median)),
        'mahal_mid10_median': float(np.mean(grid_median[mid_start:mid_end, :])),
    }

    del X_heads
    elapsed = time.time() - t0
    excl = f' [excl {exclude_field}]' if exclude_field else ''
    print(f"    {topic_name}: all={result['mahal_all']:.3f} mid10={result['mahal_mid10']:.3f} "
          f"(n={len(df_sampled)}, D={n_dem}, R={n_rep}, {elapsed:.1f}s){excl}", flush=True)

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    group = args.group
    model_configs = MODEL_GROUPS[group]

    print(f"{'='*80}")
    print(f"EXP 0A/0B DEMOGRAPHIC — GROUP {group}")
    print(f"Models: {list(model_configs.keys())}")
    print(f"Config: demo_fields=all, prompt_fmt=B, sample_frac={SAMPLE_FRAC}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    np.random.seed(42)
    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load demographic field labels
    demo_df = pd.read_csv(str(DEMO_CSV))
    all_labels = dict(zip(demo_df['VariableName'], demo_df['ConciseDescription']))
    active_fields = sorted(DEMO_VARS_ALL_YEARS)
    for f in active_fields:
        if f not in all_labels:
            all_labels[f] = f
    active_fields_set = set(active_fields)
    print(f"Active demographic fields: {len(active_fields)}")

    # Load GSS data (heavy, done once)
    df_dr, demo_parts = load_gss_data(active_fields, all_labels)

    # Load topics
    topics, pol_pub = load_filtered_topics()
    topic_list = list(topics.items())
    n_topics = len(topic_list)
    print(f"Loaded {n_topics} filtered public topics with polarization data")

    all_results = []

    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({config['type']})")
        print(f"{'='*60}")

        model, tokenizer = load_model(config["path"])
        model_type = config["type"]

        # Disable chat template for base models
        if model_type == "base":
            tokenizer.chat_template = None
            print("  Disabled chat template for base model")

        # System message
        if model_type == "base":
            system_msg = ""
        else:
            override = config.get("system_override")
            if override:
                # SmolLM3: prepend mode token to demographic system msg
                system_msg = f"{override}\n{SYSTEM_MSG_DEMO}"
            else:
                system_msg = SYSTEM_MSG_DEMO

        print(f"  System msg: {repr(system_msg[:80])}")

        model_start = time.time()

        for i, (topic_name, survey_q) in enumerate(topic_list):
            try:
                result = run_topic(
                    model, tokenizer, topic_name, survey_q,
                    df_dr, demo_parts, active_fields_set,
                    system_msg=system_msg,
                    batch_size=config["batch_size"],
                )
                if result is not None:
                    result['model'] = model_name
                    result['model_type'] = model_type
                    result['family'] = config['family']

                    # Attach GSS polarization
                    pol_row = pol_pub[pol_pub['variable'] == topic_name]
                    result['gss_polarization'] = (
                        pol_row['polarization'].values[0] if len(pol_row) > 0 else np.nan
                    )
                    all_results.append(result)

            except Exception as e:
                print(f"    ERROR on {topic_name}: {e}", flush=True)

            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - model_start
        print(f"\n  Model {model_name} total: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Save per-model checkpoint
        df_ckpt = pd.DataFrame(all_results)
        ckpt_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{group}_{model_name}_{timestamp}.pkl"
        df_ckpt.to_pickle(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.name}")

    # ==========================================================================
    # Save final results
    # ==========================================================================
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{group}_{timestamp}.csv"
    pkl_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{group}_{timestamp}.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"\nSaved: {csv_path.name}")

    # ==========================================================================
    # Analysis
    # ==========================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS — DEMOGRAPHIC: ALL-LAYER vs MID-10%")
    print(f"{'='*80}")

    print(f"\n{'Model':30s} {'Metric':12s} {'r':>7s} {'p':>8s} {'rho':>7s} {'p_rho':>8s} {'n':>4s}")
    print("-" * 85)

    for model in df['model'].unique():
        sub = df[df['model'] == model].dropna(subset=['gss_polarization'])
        if len(sub) >= 5:
            for metric, label in [('mahal_all', 'all-layer'), ('mahal_mid10', 'mid-10%'),
                                  ('mahal_all_median', 'all-median'), ('mahal_mid10_median', 'mid10-median')]:
                r, p = pearsonr(sub[metric], sub['gss_polarization'])
                rho, p_rho = spearmanr(sub[metric], sub['gss_polarization'])
                print(f"{model:30s} {label:12s} {r:7.3f} {p:8.4f} {rho:7.3f} {p_rho:8.4f} {len(sub):4d}")
            print()

    # Summary: best metric per model
    print(f"\n--- Best metric per model ---")
    for model in df['model'].unique():
        sub = df[df['model'] == model].dropna(subset=['gss_polarization'])
        if len(sub) < 5:
            continue
        best_r = -1
        best_label = ""
        for metric, label in [('mahal_all', 'all-mean'), ('mahal_mid10', 'mid10-mean'),
                              ('mahal_all_median', 'all-median'), ('mahal_mid10_median', 'mid10-median')]:
            r, _ = pearsonr(sub[metric], sub['gss_polarization'])
            if r > best_r:
                best_r = r
                best_label = label
        print(f"  {model:30s}: best={best_label} r={best_r:.3f}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 0A/0B DEMOGRAPHIC COMPLETE — GROUP {group}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
