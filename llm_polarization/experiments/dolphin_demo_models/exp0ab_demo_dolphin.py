"""
Experiment 0AB — Dolphin Instruct Models (Demographic Simulation)

Runs the GSS demographic simulation for Dolphin fine-tuned models:
  Group 1: cognitivecomputations/dolphin-2.9.1-yi-1.5-34b        (34B)
  Group 2: cognitivecomputations/dolphin-2.9.3-llama-3-8b         (8B)
  Group 3: cognitivecomputations/dolphin-2.9.3-mistral-7B-32k     (7B)
  Group 4: cognitivecomputations/Dolphin3.0-Qwen2.5-3b            (3B)
  Group 5: cognitivecomputations/dolphin-2.9.4-gemma2-2b          (2B)

For each model, processes both public + private topics across 3 conditions:
  - rhetorical: "Generate a statement by this person on {topic_desc}."
  - stance:     "What is this person's position on {topic_desc}?"
  - survey:     "If asked in a national survey about {question_text}, how would this person respond?"

Same 83-variable demographic profiles, stratified sampling (frac=0.1),
and per-head Mahalanobis grid as exp0ab_demo.py.

Models are downloaded from HuggingFace Hub if not found locally.

Usage:
    python exp0ab_demo_dolphin.py --group 1          # single group (backward compat)
    python exp0ab_demo_dolphin.py --groups 2 3 4 5   # multiple groups, one GSS load
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import gc
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from run_gss_pca import compute_all_head_metrics_pca

# =============================================================================
# Paths
# =============================================================================

DATA_ROOT   = Path("/project/jevans/maxzhuyt/gss_polarization")
MODELS_DIR  = Path("/project/jevans/maxzhuyt/models")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GSS_CSV       = DATA_ROOT / "data" / "gss" / "gss_2021_2024.csv"
STATA_2024    = DATA_ROOT / "data" / "gss" / "GSS2024.dta"
STATA_2022    = DATA_ROOT / "data" / "gss" / "GSS2022.dta"
DEMO_CSV      = DATA_ROOT / "question_lists" / "gss_demographic_variables.csv"
PUBLIC_TOPICS = DATA_ROOT / "question_lists" / "public_issues.csv"
PUBLIC_POL    = DATA_ROOT / "data" / "polarization" / "public_issues_polarization.csv"
PRIVATE_TOPICS = DATA_ROOT / "question_lists" / "private_life.csv"
PRIVATE_POL   = DATA_ROOT / "data" / "polarization" / "private_life_polarization.csv"

EXPERIMENT_NAME = "exp0ab_demo_dolphin"
SYSTEM_MSG      = "You are simulating the views of an American."
SAMPLE_FRAC     = 0.1
SAMPLE_SEED     = 42
MAX_LENGTH      = 512
PCA_DIM         = 15

# =============================================================================
# Excluded topics (same as run_demo_sim.py defaults)
# =============================================================================

EXCLUDED_PUBLIC = [
    "hubbywk1", "racdif1", "racdif2", "racdif3", "racdif4",
    "workwhts", "wlthwhts", "intlwhts",
]

EXCLUDED_PRIVATE = [
    "reborn", "marwht", "helpful", "helpfulnv", "helpfulv",
]

# =============================================================================
# Prompt conditions
# =============================================================================

CONDITIONS = ["rhetorical", "stance", "survey"]


def build_prompt(profile: str, topic_desc: str, question_text: str,
                 condition: str) -> str:
    """Build a demographic-simulation prompt for the given condition."""
    prefix = f"Given the following background about a person:\n{profile}\n\n"
    if condition == "rhetorical":
        return prefix + f"Generate a statement by this person on {topic_desc}."
    elif condition == "stance":
        return prefix + f"What is this person's position on {topic_desc}?"
    elif condition == "survey":
        return prefix + (f"If asked in a national survey about {question_text}, "
                         f"how would this person respond?")
    else:
        raise ValueError(f"Unknown condition: {condition}")


# =============================================================================
# Model groups
# =============================================================================
# HF repo IDs: models are downloaded to MODELS_DIR/<model_name> if not present.
# Batch sizes tuned for H100 80GB at max_length=512.

MODEL_GROUPS = {
    1: {
        "dphn-yi34b": {
            "hf_id":  "cognitivecomputations/dolphin-2.9.1-yi-1.5-34b",
            "local":  str(MODELS_DIR / "dolphin-2.9.1-yi-1.5-34b"),
            "batch_size": 12,   # H100-80GB: ~68GB weights, ~12GB headroom
            "family": "dphn-yi34b",
        },
    },
    2: {
        "dphn-llama8b": {
            "hf_id":  "cognitivecomputations/dolphin-2.9.3-llama-3-8b",
            "local":  str(MODELS_DIR / "dolphin-2.9.3-llama-3-8b"),
            "batch_size": 120,  # H100-80GB: ~16GB weights; A100-40GB: ~22GB peak, OK
            "family": "dphn-llama8b",
        },
    },
    3: {
        "dphn-mistral7b": {
            "hf_id":  "cognitivecomputations/dolphin-2.9.3-mistral-7B-32k",
            "local":  str(MODELS_DIR / "dolphin-2.9.3-mistral-7B-32k"),
            "batch_size": 120,  # H100-80GB: ~14GB weights; A100-40GB: ~22GB peak, OK
            "family": "dphn-mistral7b",
        },
    },
    4: {
        "dphn-qwen3b": {
            "hf_id":  "cognitivecomputations/Dolphin3.0-Qwen2.5-3b",
            "local":  str(MODELS_DIR / "Dolphin3.0-Qwen2.5-3b"),
            "batch_size": 200,  # H100/A100: ~6GB weights, very large headroom
            "family": "dphn-qwen3b",
        },
    },
    5: {
        "dphn-gemma2b": {
            "hf_id":  "cognitivecomputations/dolphin-2.9.4-gemma2-2b",
            "local":  str(MODELS_DIR / "dolphin-2.9.4-gemma2-2b"),
            "batch_size": 200,  # H100/A100: ~4GB weights, very large headroom
            "family": "dphn-gemma2b",
        },
    },
}

# =============================================================================
# 83 demographic variables (all GSS years)
# =============================================================================

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


# =============================================================================
# Model download helper
# =============================================================================

def ensure_model_local(hf_id: str, local_path: str) -> str:
    """Download model from HF Hub to local_path if not already present."""
    local = Path(local_path)
    if local.exists() and any(local.iterdir()):
        print(f"  Model already local: {local_path}")
        return local_path

    print(f"  Downloading {hf_id} → {local_path} ...")
    from huggingface_hub import snapshot_download
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf_id, local_dir=str(local))
    print(f"  Download complete: {local_path}")
    return local_path


# =============================================================================
# GSS data loading (identical to exp0ab_demo.py)
# =============================================================================

def build_code_maps(all_fields):
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
                return {int(k): str(v).strip() for k, v in vl_22[lbl].items()
                        if int(k) < max_code}
        return {}

    code_maps = {field: _build_one(field) for field in all_fields}
    del df_num_24, df_cat_24, df_num_22
    gc.collect()
    return code_maps


def precompute_demo_parts(df, code_maps, all_fields, all_labels):
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
    print("Loading GSS data...")
    df_gss = pd.read_csv(str(GSS_CSV), low_memory=False)
    print(f"  Respondents: {df_gss.shape[0]}, Variables: {df_gss.shape[1]}")

    df_gss['party_code'] = np.nan
    df_gss.loc[df_gss['partyid'].isin([0, 1, 2]), 'party_code'] = 100
    df_gss.loc[df_gss['partyid'].isin([4, 5, 6]), 'party_code'] = 200
    df_dr = df_gss[df_gss['party_code'].notna()].copy()
    n_dem = int((df_dr['party_code'] == 100).sum())
    n_rep = int((df_dr['party_code'] == 200).sum())
    print(f"  D/R respondents: {len(df_dr)} (D={n_dem}, R={n_rep})")

    df_dr['age_bin'] = pd.cut(df_dr['age'], bins=[0, 30, 40, 50, 60, 70, 100],
                               labels=False)
    strat_cols = [c for c in STRATIFY_COLS if c in df_dr.columns]
    df_dr['_strat_key'] = ''
    for col in strat_cols:
        df_dr['_strat_key'] += df_dr[col].fillna(-1).astype(int).astype(str) + '_'
    print(f"  Stratification groups: {df_dr['_strat_key'].nunique()}")

    code_maps = build_code_maps(active_fields)
    print("Pre-computing demographic profiles...")
    demo_parts = precompute_demo_parts(df_dr, code_maps, active_fields, all_labels)
    print(f"  Done: {len(demo_parts)} respondents")
    return df_dr, demo_parts


# =============================================================================
# Topic loading
# =============================================================================

def load_topics():
    """Load public + private topics with their natural-language and question text."""
    # ── public ────────────────────────────────────────────────────────────────
    pub_df  = pd.read_csv(str(PUBLIC_TOPICS))
    pol_pub = pd.read_csv(str(PUBLIC_POL))
    pol_pub_valid = set(pol_pub['variable'])
    excluded_pub_lower = {e.lower() for e in EXCLUDED_PUBLIC}

    public = {}
    for _, row in pub_df.iterrows():
        var = str(row['Variable']).strip()
        if var.lower() in excluded_pub_lower:
            continue
        if var not in pol_pub_valid:
            continue
        topic_desc   = str(row.get('NaturalLanguageClause', '') or '').strip()
        question_text = str(row.get('SurveyQuestion', '') or '').strip()
        if not topic_desc:
            topic_desc = question_text  # fallback
        public[var] = {'topic_desc': topic_desc, 'question_text': question_text,
                       'category': 'public'}

    # ── private ───────────────────────────────────────────────────────────────
    prv_df  = pd.read_csv(str(PRIVATE_TOPICS))
    pol_prv = pd.read_csv(str(PRIVATE_POL))
    pol_prv_valid = set(pol_prv['variable'])
    excluded_prv_lower = {e.lower() for e in EXCLUDED_PRIVATE}

    private = {}
    for _, row in prv_df.iterrows():
        var = str(row['Variable']).strip()
        if var.lower() in excluded_prv_lower:
            continue
        if var not in pol_prv_valid:
            continue
        topic_desc   = str(row.get('NaturalLanguageClause', '') or '').strip()
        question_text = str(row.get('SurveyQuestion', '') or '').strip()
        if not topic_desc:
            topic_desc = question_text
        private[var] = {'topic_desc': topic_desc, 'question_text': question_text,
                        'category': 'private'}

    # polarization lookup (both categories)
    pol_lookup = {}
    for _, r in pol_pub.iterrows():
        pol_lookup[str(r['variable'])] = float(r['polarization'])
    for _, r in pol_prv.iterrows():
        pol_lookup[str(r['variable'])] = float(r['polarization'])

    print(f"Topics loaded: {len(public)} public, {len(private)} private")
    return public, private, pol_lookup


# =============================================================================
# Per-topic analysis
# =============================================================================

def run_topic(model, tokenizer, topic_name, topic_info,
              df_dr, demo_parts, active_fields_set,
              condition, batch_size):
    """
    Run one topic under one condition. Returns result dict or None.
    """
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

    exclude_field = topic_name if topic_name in active_fields_set else None

    topic_desc    = topic_info['topic_desc']
    question_text = topic_info['question_text']

    rng = np.random.default_rng(hash(topic_name + condition) % (2**32))
    prompts = []
    for idx in df_sampled.index:
        if exclude_field:
            parts = [p[1] for p in demo_parts[idx] if p[0] != exclude_field]
        else:
            parts = [p[1] for p in demo_parts[idx]]
        rng.shuffle(parts)
        profile = '. '.join(parts) + '.'
        prompts.append(build_prompt(profile, topic_desc, question_text, condition))

    labels = df_sampled['party_code'].values.astype(int)

    X_heads = extract_heads_batched(
        model, tokenizer, prompts, SYSTEM_MSG,
        batch_size=batch_size, max_length=MAX_LENGTH,
    )

    N, L, H, D = X_heads.shape

    grid_mean   = compute_all_head_metrics_pca(
        X_heads, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='mean')
    grid_median = compute_all_head_metrics_pca(
        X_heads, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='median')

    mid_start = int(L * 0.45)
    mid_end   = max(int(L * 0.55), mid_start + 1)

    result = {
        'topic':              topic_name,
        'category':           topic_info['category'],
        'condition':          condition,
        'n_sampled':          len(df_sampled),
        'n_dem':              n_dem,
        'n_rep':              n_rep,
        'n_layers':           L,
        'mid_layers':         f"{mid_start}-{mid_end-1}",
        'mahal_all':          float(np.mean(grid_mean)),
        'mahal_mid10':        float(np.mean(grid_mean[mid_start:mid_end, :])),
        'mahal_max':          float(np.max(grid_mean)),
        'mahal_all_median':   float(np.mean(grid_median)),
        'mahal_mid10_median': float(np.mean(grid_median[mid_start:mid_end, :])),
    }

    del X_heads
    elapsed = time.time() - t0
    excl = f' [excl {exclude_field}]' if exclude_field else ''
    print(f"    [{condition}] {topic_name}: all={result['mahal_all']:.3f} "
          f"mid10={result['mahal_mid10']:.3f} "
          f"(n={len(df_sampled)}, D={n_dem}, R={n_rep}, {elapsed:.1f}s){excl}",
          flush=True)
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',  type=int, choices=[1, 2, 3, 4, 5],
                        help='Single group to run (backward compatible with existing sbatch files)')
    parser.add_argument('--groups', type=int, nargs='+', choices=[1, 2, 3, 4, 5],
                        help='One or more groups to run sequentially (GSS data loaded once)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip (model, category, condition) combos that already have a checkpoint file')
    args = parser.parse_args()

    if args.groups:
        groups = args.groups
    elif args.group:
        groups = [args.group]
    else:
        parser.error('Specify --group N or --groups N [M ...]')

    groups_str = '-'.join(map(str, groups))

    print(f"{'='*80}")
    print(f"EXP 0AB DEMOGRAPHIC — DOLPHIN MODELS — GROUPS {groups}")
    print(f"Models:     {[list(MODEL_GROUPS[g].keys())[0] for g in groups]}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Categories: public + private")
    print(f"Config:     demo_fields=83, sample_frac={SAMPLE_FRAC}, max_length={MAX_LENGTH}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    np.random.seed(42)
    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load demographic field labels
    demo_df   = pd.read_csv(str(DEMO_CSV))
    all_labels = dict(zip(demo_df['VariableName'], demo_df['ConciseDescription']))
    active_fields = sorted(DEMO_VARS_ALL_YEARS)
    for f in active_fields:
        if f not in all_labels:
            all_labels[f] = f
    active_fields_set = set(active_fields)
    print(f"Active demographic fields: {len(active_fields)}")

    # Load GSS data once — shared across all groups/models/conditions
    df_dr, demo_parts = load_gss_data(active_fields, all_labels)

    # Load topics once
    public_topics, private_topics, pol_lookup = load_topics()
    all_topic_sets = [
        ('public',  public_topics),
        ('private', private_topics),
    ]

    all_results = []

    # ── Outer loop: group → model ────────────────────────────────────────────
    for group in groups:
        model_configs = MODEL_GROUPS[group]

        for model_name, config in model_configs.items():
            print(f"\n{'='*60}")
            print(f"Group {group}  |  Model: {model_name}  ({config['hf_id']})")
            print(f"{'='*60}")

            local_path = ensure_model_local(config['hf_id'], config['local'])
            model, tokenizer = load_model(local_path)
            batch_size = config['batch_size']

            model_start = time.time()

            # ── Loop: category × condition × topic ──────────────────────────
            for cat_name, topics in all_topic_sets:
                for condition in CONDITIONS:
                    # --resume: skip if a checkpoint already exists for this combo
                    if args.resume:
                        existing = sorted(RESULTS_DIR.glob(
                            f"{EXPERIMENT_NAME}_g{group}_{model_name}"
                            f"_{cat_name}_{condition}_*.pkl"
                        ))
                        if existing:
                            print(f"\n  ── {cat_name} / {condition} "
                                  f"[SKIPPED — checkpoint exists: {existing[-1].name}]")
                            # Load the most recent checkpoint to restore all_results
                            loaded = pd.read_pickle(existing[-1])
                            loaded_rows = loaded.to_dict('records')
                            # Merge: add rows not already in all_results
                            existing_keys = {
                                (r['model'], r.get('category', ''), r.get('condition', ''),
                                 r.get('topic', ''))
                                for r in all_results
                            }
                            for row in loaded_rows:
                                key = (row.get('model', ''), row.get('category', ''),
                                       row.get('condition', ''), row.get('topic', ''))
                                if key not in existing_keys:
                                    all_results.append(row)
                                    existing_keys.add(key)
                            continue

                    print(f"\n  ── {cat_name} / {condition} "
                          f"({len(topics)} topics) ──────────────────────────")
                    cond_start = time.time()
                    cond_results = []

                    for topic_name, topic_info in topics.items():
                        try:
                            result = run_topic(
                                model, tokenizer, topic_name, topic_info,
                                df_dr, demo_parts, active_fields_set,
                                condition=condition, batch_size=batch_size,
                            )
                            if result is not None:
                                result['model']  = model_name
                                result['family'] = config['family']
                                result['group']  = group
                                result['gss_polarization'] = pol_lookup.get(
                                    topic_name, np.nan)
                                cond_results.append(result)
                                all_results.append(result)
                        except Exception as e:
                            print(f"    ERROR on {topic_name}: {e}", flush=True)

                        gc.collect()
                        torch.cuda.empty_cache()

                    # Per-condition checkpoint
                    if cond_results:
                        df_ckpt = pd.DataFrame(all_results)
                        ckpt = (RESULTS_DIR /
                                f"{EXPERIMENT_NAME}_g{group}_{model_name}"
                                f"_{cat_name}_{condition}_{timestamp}.pkl")
                        df_ckpt.to_pickle(ckpt)
                        elapsed_cond = time.time() - cond_start
                        print(f"  Checkpoint saved: {ckpt.name} "
                              f"({len(cond_results)} topics, {elapsed_cond:.0f}s)")

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            elapsed_model = time.time() - model_start
            print(f"\n  Model {model_name} total: {elapsed_model:.0f}s "
                  f"({elapsed_model/3600:.1f}h)")

    # ── Save final CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{groups_str}_{timestamp}.csv"
    pkl_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{groups_str}_{timestamp}.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"\nSaved: {csv_path.name}  ({len(df)} rows)")

    print(f"\n{'='*80}")
    print(f"COMPLETE — GROUPS {groups}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
