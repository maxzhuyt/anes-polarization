"""
Large Model Experiments: Politician + Demographic Simulation

Replicates the exp0ab pipeline (prompt framing, layer selection) for large models:
  - Politician: 3 prompt conditions (rhetorical/stance/survey) × all-layer + mid-10%
  - Demographic: format B × all-layer + mid-10% (mean + median centroids)

Each job runs ONE model only (for fault isolation).

Topic sets (--topic-set):
  public   — 126 filtered public issues (default, existing behaviour)
  private  — all private-life topics (no exclusion filter)
  all      — all public + all private (no exclusion filters)

Usage:
    python exp_large_models.py --model mistral-24b
    python exp_large_models.py --model qwen3-32b
    python exp_large_models.py --model deepseek-r1-32b
    python exp_large_models.py --model dolphin-34b
    python exp_large_models.py --model qwen2.5-72b
    python exp_large_models.py --model llama-70b
    python exp_large_models.py --model glm-air
    python exp_large_models.py --model llama-70b --politician-only
    python exp_large_models.py --model llama-70b --demo-only
    python exp_large_models.py --model mistral-24b --topic-set private --politician-only
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import gc
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import set_random_seeds, RESULTS_DIR, load_polarization_data

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import extract_heads_batched
from run_gss_pca import compute_all_head_metrics_pca

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp_large"
MODELS_DIR = "/project/jevans/maxzhuyt/models"
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
QUESTION_LISTS_DIR = Path("/project/jevans/maxzhuyt/gss_polarization/question_lists")

# Paths for demographic simulation
DATA_ROOT = Path("/project/jevans/maxzhuyt/gss_polarization")
GSS_CSV = DATA_ROOT / "data" / "gss" / "gss_2021_2024.csv"
STATA_2024 = DATA_ROOT / "data" / "gss" / "GSS2024.dta"
STATA_2022 = DATA_ROOT / "data" / "gss" / "GSS2022.dta"
DEMO_CSV = DATA_ROOT / "question_lists" / "gss_demographic_variables.csv"
PUBLIC_POL_CSV = DATA_ROOT / "data" / "polarization" / "public_issues_polarization.csv"

PCA_DIM = 15
POL_MAX_LENGTH = 128
DEMO_MAX_LENGTH = 512
DEMO_SAMPLE_FRAC = 0.1
DEMO_SAMPLE_SEED = 42
DEMO_SYSTEM_MSG = "You are simulating the views of an American."
DEMO_PROMPT_FMT = "Given the following background about a person:\n{profile}\n\nHow would they answer: {question}"

EXCLUDED_PUBLIC = [
    "hubbywk1", "racdif1", "racdif2", "racdif3", "racdif4",
    "workwhts", "wlthwhts", "intlwhts",
]

SYSTEM_MSG_POLITICIAN = "You are simulating the public stance of U.S. politicians.\n\n"


# =============================================================================
# Model loading (with optional quantization for very large models)
# =============================================================================

def load_model_large(path: str, quantize: bool = False):
    """
    Load model with optional 4-bit quantization for very large models.
    Uses bitsandbytes NF4 quantization when quantize=True.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading model from: {path}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = dict(
        device_map="auto",
        local_files_only=True,
        attn_implementation="sdpa",
    )

    if quantize:
        print("  Using 4-bit NF4 quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

CONDITIONS = ["rhetorical", "stance", "survey"]

STRATIFY_COLS = ['polviews', 'age_bin', 'degree', 'race', 'sex', 'rincome']

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

# =============================================================================
# Model configs — one entry per model
# =============================================================================

MODEL_CONFIGS = {
    "mistral-24b": {
        "name": "Mistral-Small-24B",
        "path": f"{MODELS_DIR}/Mistral-Small-24B-Instruct-2501",
        "type": "instruct",
        "family": "Mistral-Small-24B",
        "pol_batch_size": 64,
        "demo_batch_size": 8,
    },
    "qwen3-32b": {
        "name": "Qwen3-32B",
        "path": f"{MODELS_DIR}/Qwen3-32B",
        "type": "instruct",
        "family": "Qwen3-32B",
        "pol_batch_size": 40,
        "demo_batch_size": 6,
    },
    "deepseek-r1-32b": {
        "name": "DeepSeek-R1-Distill-Qwen-32B",
        "path": f"{MODELS_DIR}/DeepSeek-R1-Distill-Qwen-32B",
        "type": "reasoning",
        "family": "DeepSeek-R1-32B",
        "pol_batch_size": 40,
        "demo_batch_size": 6,
    },
    "dolphin-34b": {
        "name": "dolphin-2.9.1-yi-1.5-34b",
        "path": f"{MODELS_DIR}/dolphin-2.9.1-yi-1.5-34b",
        "type": "instruct",
        "family": "dolphin-yi-34B",
        "pol_batch_size": 40,
        "demo_batch_size": 6,
    },
    "qwen2.5-72b": {
        "name": "Qwen2.5-72B-Instruct",
        "path": f"{MODELS_DIR}/Qwen2.5-72B-Instruct",
        "type": "instruct",
        "family": "Qwen2.5-72B",
        "pol_batch_size": 16,
        "demo_batch_size": 2,
    },
    "llama-70b": {
        "name": "Llama-3.3-70B-Instruct",
        "path": f"{MODELS_DIR}/Llama-3.3-70B-Instruct",
        "type": "instruct",
        "family": "Llama-3.3-70B",
        "pol_batch_size": 16,
        "demo_batch_size": 2,
    },
    "glm-air": {
        "name": "GLM-4.5-Air",
        "path": f"{MODELS_DIR}/GLM-4.5-Air",
        "type": "instruct",
        "family": "GLM-4.5-Air",
        "pol_batch_size": 8,
        "demo_batch_size": 2,
    },
}


# =============================================================================
# Topic loading
# =============================================================================

def load_filtered_public_topics() -> Dict[str, str]:
    """Load 126 filtered public issue topics."""
    csv_path = QUESTION_LISTS_DIR / "public_issues.csv"
    df = pd.read_csv(csv_path)
    topics = {}
    for _, row in df.iterrows():
        var = str(row['Variable']).lower()
        if var not in EXCLUDED_PUBLIC:
            topics[var] = str(row['NaturalLanguageClause']).strip()
    return topics


def load_all_public_topics() -> Dict[str, str]:
    """Load all public issue topics with no exclusion filter."""
    csv_path = QUESTION_LISTS_DIR / "public_issues.csv"
    df = pd.read_csv(csv_path)
    topics = {}
    for _, row in df.iterrows():
        var = str(row['Variable']).lower()
        nlc = str(row.get('NaturalLanguageClause', '')).strip()
        if nlc and nlc != 'nan':
            topics[var] = nlc
    return topics


def load_all_private_topics() -> Dict[str, str]:
    """Load all private-life topics with no exclusion filter."""
    csv_path = QUESTION_LISTS_DIR / "private_life.csv"
    df = pd.read_csv(csv_path)
    topics = {}
    for _, row in df.iterrows():
        var = str(row['Variable']).lower()
        nlc = str(row.get('NaturalLanguageClause', '')).strip()
        if not nlc or nlc == 'nan':
            nlc = str(row.get('SurveyQuestion', '')).strip()
        if nlc and nlc != 'nan':
            topics[var] = nlc
    return topics


def build_topic_list(topic_set: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (variable, nlc, category) tuples.

    topic_set:
        'public'  — filtered public topics
        'private' — all private topics (no exclusion filter)
        'all'     — all public + all private (no exclusion filters)
    """
    items = []
    if topic_set == 'public':
        for v, d in load_filtered_public_topics().items():
            items.append((v, d, 'public'))
    elif topic_set == 'private':
        for v, d in load_all_private_topics().items():
            items.append((v, d, 'private'))
    elif topic_set == 'all':
        for v, d in load_all_public_topics().items():
            items.append((v, d, 'public'))
        for v, d in load_all_private_topics().items():
            items.append((v, d, 'private'))
    else:
        raise ValueError(f"Unknown topic_set: {topic_set}")
    return items


def load_question_text() -> Dict[str, str]:
    """Load NLC / survey text for survey-style prompts (both public + private)."""
    q_map = {}
    for csv_name in ["public_issues.csv", "private_life.csv"]:
        csv_path = QUESTION_LISTS_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                var = str(row['Variable']).lower()
                nlc = str(row.get('NaturalLanguageClause', '')).strip()
                if nlc and nlc != 'nan':
                    q_map[var] = nlc
                elif pd.notna(row.get('SurveyQuestion', None)):
                    q_map[var] = str(row['SurveyQuestion']).strip()
    return q_map


# =============================================================================
# Prompt generation (politician)
# =============================================================================

def generate_prompts(topic_desc: str, question_text: str,
                     politician_names: List[str], model_type: str,
                     condition: str) -> List[str]:
    """Generate prompts for a specific condition and model type."""
    prompts = []
    for name in politician_names:
        if condition == "rhetorical":
            if model_type == "base":
                prompt = f"{name} makes a statement on {topic_desc}:"
            else:
                prompt = f"Generate a statement by {name} on {topic_desc}."
        elif condition == "stance":
            if model_type == "base":
                prompt = f"On {topic_desc}, {name}'s position is"
            else:
                prompt = f"What is {name}'s position on {topic_desc}?"
        elif condition == "survey":
            if model_type == "base":
                prompt = f"In a survey about {question_text}, {name} would respond"
            else:
                prompt = f"If asked in a national survey about {question_text}, how would {name} respond?"
        else:
            raise ValueError(f"Unknown condition: {condition}")
        prompts.append(prompt)
    return prompts


# =============================================================================
# Per-head Mahalanobis computation
# =============================================================================

def compute_both_metrics(activations: np.ndarray, labels: np.ndarray) -> dict:
    """Per-head Mahalanobis grid -> all-layer and mid-10% metrics."""
    N, L, H, D = activations.shape

    grid = compute_all_head_metrics_pca(
        activations, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='mean',
    )

    mid_start = int(L * 0.45)
    mid_end = max(int(L * 0.55), mid_start + 1)

    return {
        'mahal_all': float(np.mean(grid)),
        'mahal_mid10': float(np.mean(grid[mid_start:mid_end, :])),
        'n_layers': L,
        'mid_layers': f"{mid_start}-{mid_end-1}",
        'n_mid_layers': mid_end - mid_start,
    }


def compute_demo_metrics(activations: np.ndarray, labels: np.ndarray) -> dict:
    """Per-head Mahalanobis for demographic sim (mean + median centroids)."""
    N, L, H, D = activations.shape

    grid_mean = compute_all_head_metrics_pca(
        activations, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='mean',
    )
    grid_median = compute_all_head_metrics_pca(
        activations, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='median',
    )

    mid_start = int(L * 0.45)
    mid_end = max(int(L * 0.55), mid_start + 1)

    return {
        'n_layers': L,
        'mid_layers': f"{mid_start}-{mid_end-1}",
        'mahal_all': float(np.mean(grid_mean)),
        'mahal_mid10': float(np.mean(grid_mean[mid_start:mid_end, :])),
        'mahal_max': float(np.max(grid_mean)),
        'mahal_all_median': float(np.mean(grid_median)),
        'mahal_mid10_median': float(np.mean(grid_median[mid_start:mid_end, :])),
    }


# =============================================================================
# Demographic simulation helpers (from exp0ab_demo.py)
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

    df_dr['age_bin'] = pd.cut(df_dr['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=False)
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


def load_filtered_topics_for_demo():
    """Load 126 filtered public topics that have polarization data."""
    pub_topics = pd.read_csv(str(QUESTION_LISTS_DIR / "public_issues.csv"))
    pol_pub = pd.read_csv(str(PUBLIC_POL_CSV))
    pol_valid = set(pol_pub['variable'])
    topics = {}
    for _, row in pub_topics.iterrows():
        var = row['Variable']
        if var in pol_valid and var.lower() not in [e.lower() for e in EXCLUDED_PUBLIC]:
            topics[var] = row['SurveyQuestion']
    return topics, pol_pub


# =============================================================================
# Politician simulation
# =============================================================================

def run_politician_simulation(model, tokenizer, config, topics, q_text,
                              politician_names, party_labels, pol_data,
                              timestamp, topic_tag="pol"):
    """
    Run 3-condition politician simulation for a single loaded model.

    topic_tag: label used in output filenames
               ('pol' for public, 'private_pol' for private, 'all_pol' for all).
    """
    model_name = config["name"]
    model_type = config["type"]
    batch_size = config["pol_batch_size"]

    if model_type == "base":
        system_msg = ""
    else:
        system_msg = SYSTEM_MSG_POLITICIAN

    topic_list = list(topics.items())
    n_topics = len(topic_list)
    all_results = []

    for cond in CONDITIONS:
        print(f"\n  --- Condition: {cond} ---")
        cond_start = time.time()

        for i, (topic, topic_desc) in enumerate(topic_list):
            question = q_text.get(topic, topic_desc)

            prompts = generate_prompts(
                topic_desc, question, politician_names, model_type, cond
            )

            activations = extract_heads_batched(
                model, tokenizer, prompts, system_msg,
                batch_size=batch_size, max_length=POL_MAX_LENGTH,
            )

            metrics = compute_both_metrics(activations, party_labels)

            pol_row = pol_data[pol_data['variable'] == topic]
            gss_pol = pol_row['polarization'].values[0] if len(pol_row) > 0 else np.nan

            all_results.append({
                'model': model_name,
                'model_type': model_type,
                'family': config['family'],
                'condition': cond,
                'topic': topic,
                'mahal_all': metrics['mahal_all'],
                'mahal_mid10': metrics['mahal_mid10'],
                'n_layers': metrics['n_layers'],
                'mid_layers': metrics['mid_layers'],
                'gss_polarization': gss_pol,
            })

            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1}/{n_topics}] {topic:15s} all={metrics['mahal_all']:.3f} "
                      f"mid10={metrics['mahal_mid10']:.3f} "
                      f"gss={'nan' if np.isnan(gss_pol) else f'{gss_pol:.3f}'}")

            del activations
            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - cond_start
        print(f"  Condition {cond} done in {elapsed:.0f}s ({elapsed/n_topics:.1f}s/topic)")

        # Save per-condition checkpoint
        df_ckpt = pd.DataFrame(all_results)
        ckpt_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_{topic_tag}_{model_name}_{cond}_{timestamp}.pkl"
        df_ckpt.to_pickle(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.name}")

    return all_results


# =============================================================================
# Demographic simulation
# =============================================================================

def run_demo_topic(model, tokenizer, topic_name, survey_question,
                   df_dr, demo_parts, active_fields_set,
                   system_msg, batch_size):
    """Run a single demographic topic."""
    t0 = time.time()

    if topic_name not in df_dr.columns:
        return None

    valid_mask = df_dr[topic_name].apply(is_valid_response)
    df_valid = df_dr[valid_mask]
    if len(df_valid) < 20:
        return None

    df_sampled = stratified_sample(df_valid, DEMO_SAMPLE_FRAC, DEMO_SAMPLE_SEED)
    n_dem = int((df_sampled['party_code'] == 100).sum())
    n_rep = int((df_sampled['party_code'] == 200).sum())
    if n_dem < 5 or n_rep < 5:
        return None

    exclude_field = topic_name if topic_name in active_fields_set else None

    rng = np.random.default_rng(hash(topic_name) % (2**32))
    prompts = []
    for idx in df_sampled.index:
        if exclude_field:
            parts = [p[1] for p in demo_parts[idx] if p[0] != exclude_field]
        else:
            parts = [p[1] for p in demo_parts[idx]]
        rng.shuffle(parts)
        profile = '. '.join(parts) + '.'
        prompts.append(DEMO_PROMPT_FMT.format(profile=profile, question=survey_question))

    labels = df_sampled['party_code'].values.astype(int)

    X_heads = extract_heads_batched(
        model, tokenizer, prompts, system_msg,
        batch_size=batch_size, max_length=DEMO_MAX_LENGTH,
    )

    metrics = compute_demo_metrics(X_heads, labels)
    metrics['topic'] = topic_name
    metrics['n_sampled'] = len(df_sampled)
    metrics['n_dem'] = n_dem
    metrics['n_rep'] = n_rep

    del X_heads
    elapsed = time.time() - t0
    excl = f' [excl {exclude_field}]' if exclude_field else ''
    print(f"    {topic_name}: all={metrics['mahal_all']:.3f} mid10={metrics['mahal_mid10']:.3f} "
          f"(n={len(df_sampled)}, D={n_dem}, R={n_rep}, {elapsed:.1f}s){excl}", flush=True)

    return metrics


def run_demographic_simulation(model, tokenizer, config, timestamp):
    """Run demographic simulation for a single loaded model."""
    model_name = config["name"]
    model_type = config["type"]
    batch_size = config["demo_batch_size"]

    # Load demo data
    demo_df = pd.read_csv(str(DEMO_CSV))
    all_labels = dict(zip(demo_df['VariableName'], demo_df['ConciseDescription']))
    active_fields = sorted(DEMO_VARS_ALL_YEARS)
    for f in active_fields:
        if f not in all_labels:
            all_labels[f] = f
    active_fields_set = set(active_fields)
    print(f"Active demographic fields: {len(active_fields)}")

    df_dr, demo_parts = load_gss_data(active_fields, all_labels)

    topics, pol_pub = load_filtered_topics_for_demo()
    topic_list = list(topics.items())
    n_topics = len(topic_list)
    print(f"Loaded {n_topics} filtered public topics")

    if model_type == "base":
        system_msg = ""
    else:
        system_msg = DEMO_SYSTEM_MSG

    all_results = []

    for i, (topic_name, survey_q) in enumerate(topic_list):
        try:
            result = run_demo_topic(
                model, tokenizer, topic_name, survey_q,
                df_dr, demo_parts, active_fields_set,
                system_msg=system_msg,
                batch_size=batch_size,
            )
            if result is not None:
                result['model'] = model_name
                result['model_type'] = model_type
                result['family'] = config['family']

                pol_row = pol_pub[pol_pub['variable'] == topic_name]
                result['gss_polarization'] = (
                    pol_row['polarization'].values[0] if len(pol_row) > 0 else np.nan
                )
                all_results.append(result)

        except Exception as e:
            print(f"    ERROR on {topic_name}: {e}", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

        # Periodic checkpoint every 25 topics
        if (i + 1) % 25 == 0:
            df_ckpt = pd.DataFrame(all_results)
            ckpt_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_demo_{model_name}_ckpt{i+1}_{timestamp}.pkl"
            df_ckpt.to_pickle(ckpt_path)
            print(f"  Saved checkpoint ({i+1}/{n_topics}): {ckpt_path.name}")

    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--politician-only', action='store_true',
                        help='Run only politician simulation')
    parser.add_argument('--demo-only', action='store_true',
                        help='Run only demographic simulation')
    parser.add_argument('--quantize', action='store_true',
                        help='Load model with 4-bit NF4 quantization (for very large models)')
    parser.add_argument(
        '--topic-set', type=str, default='public',
        choices=['public', 'private', 'all'],
        help=(
            "Topic set for politician simulation: "
            "'public' = filtered public topics (default); "
            "'private' = all private-life topics incl. excluded; "
            "'all' = all public + all private. "
            "Note: demographic simulation always uses public topics only."
        ),
    )
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    run_politician = not args.demo_only
    # Demographic simulation is only meaningful on public topics
    run_demo = not args.politician_only and args.topic_set == 'public'

    # Output filename tag that encodes the topic set
    topic_tag = "pol" if args.topic_set == 'public' else f"{args.topic_set}_pol"

    print(f"{'='*80}")
    print(f"LARGE MODEL EXPERIMENT — {config['name']}")
    print(f"Type: {config['type']}, Family: {config['family']}")
    print(f"Topic set: {args.topic_set}")
    print(f"Politician: {'YES' if run_politician else 'SKIP'} (batch={config['pol_batch_size']})")
    print(f"Demographic: {'YES' if run_demo else 'SKIP'} (batch={config['demo_batch_size']})")
    print(f"Quantize: {args.quantize}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    set_random_seeds(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model once, use for both simulations
    print(f"\nLoading model from {config['path']}...")
    t_load = time.time()
    model, tokenizer = load_model_large(config["path"], quantize=args.quantize)
    print(f"Model loaded in {time.time() - t_load:.0f}s")

    if config["type"] == "base":
        tokenizer.chat_template = None
        print("Disabled chat template for base model")

    # =========================================================================
    # Part 1: Politician simulation
    # =========================================================================
    pol_results = []
    if run_politician:
        print(f"\n{'='*80}")
        print(f"PART 1: POLITICIAN SIMULATION  (topic_set={args.topic_set})")
        print(f"{'='*80}")

        topic_items = build_topic_list(args.topic_set)
        q_text = load_question_text()
        topics = {v: d for v, d, _ in topic_items}
        n_pub = sum(1 for _, _, c in topic_items if c == 'public')
        n_prv = sum(1 for _, _, c in topic_items if c == 'private')
        print(f"Loaded {len(topics)} topics (public={n_pub}, private={n_prv})")

        df_pol = pd.read_csv(POLITICIAN_CSV)
        df_pol = df_pol[df_pol['party_code'].isin([100, 200])].dropna(subset=['fullname'])
        politician_names = df_pol['fullname'].tolist()
        party_labels = df_pol['party_code'].values
        print(f"Loaded {len(politician_names)} politicians")

        pol_data = load_polarization_data()
        print(f"Loaded polarization data: {len(pol_data)} variables")

        t_pol = time.time()
        pol_results = run_politician_simulation(
            model, tokenizer, config, topics, q_text,
            politician_names, party_labels, pol_data, timestamp,
            topic_tag=topic_tag,
        )

        df_pol_results = pd.DataFrame(pol_results)
        csv_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_{topic_tag}_{config['name']}_{timestamp}.csv"
        pkl_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_{topic_tag}_{config['name']}_{timestamp}.pkl"
        df_pol_results.to_csv(csv_path, index=False)
        df_pol_results.to_pickle(pkl_path)
        print(f"\nPolitician results saved: {csv_path.name}")
        print(f"Politician simulation took: {(time.time() - t_pol)/60:.1f} min")

    # =========================================================================
    # Part 2: Demographic simulation (public topics only)
    # =========================================================================
    demo_results = []
    if run_demo:
        print(f"\n{'='*80}")
        print("PART 2: DEMOGRAPHIC SIMULATION")
        print(f"{'='*80}")

        t_demo = time.time()
        demo_results = run_demographic_simulation(
            model, tokenizer, config, timestamp
        )

        df_demo_results = pd.DataFrame(demo_results)
        csv_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_demo_{config['name']}_{timestamp}.csv"
        pkl_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_demo_{config['name']}_{timestamp}.pkl"
        df_demo_results.to_csv(csv_path, index=False)
        df_demo_results.to_pickle(pkl_path)
        print(f"\nDemographic results saved: {csv_path.name}")
        print(f"Demographic simulation took: {(time.time() - t_demo)/60:.1f} min")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"LARGE MODEL EXPERIMENT COMPLETE — {config['name']}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
