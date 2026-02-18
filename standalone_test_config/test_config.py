#!/usr/bin/env python3
"""
Standalone fast configuration testing for demographic simulation.
Designed for a single RTX 3090 (24GB VRAM).

Self-contained: no external imports from the project.
Models are loaded from HuggingFace.

Usage:
    python test_config.py --model-id meta-llama/Llama-3.1-8B-Instruct --model-name Llama-3.1-8B-Instruct
    python test_config.py --model-id google/gemma-2-9b-it --model-name gemma-2-9b-it
    python test_config.py --demo-fields core --prompt-fmt A
    python test_config.py --help
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import pandas as pd
import numpy as np
import torch
import gc
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, LinAlgError
from joblib import Parallel, delayed
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

SYSTEM_MSG = 'You are simulating the views of an American.'

DATA_DIR = Path(__file__).parent / 'data'

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

LIFESTYLE_VARS_ALL_YEARS = {
    'attend', 'compuse', 'fear', 'helpoth', 'hivtest', 'hunt', 'news', 'obey',
    'owngun', 'partners', 'pistol', 'popular', 'pray', 'reborn', 'relactiv',
    'rifle', 'rowngun', 'sexfreq', 'shotgun', 'socbar', 'socfrend', 'socommun',
    'socrel', 'spanking', 'thnkself', 'tvhours', 'union', 'vetyears', 'vote16',
    'webmob', 'workhard', 'wrkgovt1', 'wrkslf', 'xmovie',
}

DEMO_FIELDS_CORE = [
    'age', 'sex', 'race', 'degree', 'marital', 'relig', 'region',
    'rincome', 'wrkstat', 'childs', 'polviews',
]

DEMO_FIELDS_EXPANDED = [
    'age', 'sex', 'race', 'degree', 'marital', 'relig', 'region',
    'rincome', 'wrkstat', 'childs', 'polviews', 'born', 'educ',
    'health', 'sibs', 'sexornt',
]

PROMPT_FORMATS = {
    'A': '{profile}\n\nSurvey question: {question}',
    'B': 'Given the following background about a person:\n{profile}\n\nHow would they answer: {question}',
    'C': '{profile}\n\n{question}',
}

STRATIFY_COLS = ['polviews', 'age_bin', 'degree', 'race', 'born', 'sex']


# =============================================================================
# MODEL LOADING (from HuggingFace)
# =============================================================================

def load_model(model_id: str, dtype=None, attn_implementation: str = "sdpa"):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading model: {model_id}...")

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_heads_batched(model, tokenizer, texts, system_msg,
                          batch_size=8, max_length=512):
    """Extract per-head activations from the last token position."""
    model.eval()
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    D_head = getattr(model.config, 'head_dim', model.config.hidden_size // H)

    activations = []
    layer_outputs = [None] * L

    def get_hook(layer_idx):
        def hook(module, input, output):
            reshaped = input[0].detach().view(input[0].shape[0], input[0].shape[1], H, D_head)
            layer_outputs[layer_idx] = reshaped[:, -1, :, :].float().cpu().numpy()
        return hook

    hooks = []
    for li in range(L):
        hooks.append(
            model.model.layers[li].self_attn.o_proj.register_forward_hook(get_hook(li))
        )

    def format_prompt(text):
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_msg},
                     {"role": "user", "content": text}],
                    tokenize=False, add_generation_prompt=True
                )
            except Exception:
                try:
                    combined = f"{system_msg}\n\n{text}"
                    return tokenizer.apply_chat_template(
                        [{"role": "user", "content": combined}],
                        tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
        return f"{system_msg}\n\n{text}"

    print(f"  > Extracting with Batch Size {batch_size}...", flush=True)
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            formatted_batch = [format_prompt(t) for t in batch]
            enc = tokenizer(
                formatted_batch, return_tensors="pt",
                padding=True, truncation=True, max_length=max_length
            ).to(model.device)
            model(**enc)
            batch_acts = np.stack(layer_outputs, axis=1)
            activations.append(batch_acts)
    finally:
        for h in hooks:
            h.remove()

    return np.concatenate(activations, axis=0)


def get_model_info(model):
    config = model.config
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    return {
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "head_dim": head_dim,
    }


# =============================================================================
# PCA MAHALANOBIS
# =============================================================================

def weighted_mean(X, weights):
    weights = weights / weights.sum()
    return np.sum(X * weights[:, np.newaxis], axis=0)


def weighted_median(X, weights):
    weights = weights / weights.sum()
    result = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        sorted_idx = np.argsort(X[:, j])
        sorted_vals = X[sorted_idx, j]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)
        median_idx = min(median_idx, len(sorted_vals) - 1)
        result[j] = sorted_vals[median_idx]
    return result


def weighted_cov(X, weights):
    weights = weights / weights.sum()
    mean = weighted_mean(X, weights)
    X_centered = X - mean
    cov = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(X)):
        cov += weights[i] * np.outer(X_centered[i], X_centered[i])
    sum_w = weights.sum()
    sum_w2 = (weights ** 2).sum()
    cov = cov / (1 - sum_w2 / (sum_w ** 2))
    return cov


def compute_mahalanobis_pca(head_data, group_labels, group_values=(100, 200),
                             n_components=10, centroid_method='mean'):
    valid_mask = np.isin(group_labels, group_values)
    X = head_data[valid_mask]
    y = group_labels[valid_mask]
    w = np.ones(len(X))

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    try:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
    except Exception:
        return 0.0

    try:
        mask1 = y == group_values[0]
        mask2 = y == group_values[1]
        group1, group2 = X_pca[mask1], X_pca[mask2]
        w1, w2 = w[mask1], w[mask2]

        if len(group1) > 5 and len(group2) > 5:
            if centroid_method == 'median':
                mu_1 = weighted_median(group1, w1)
                mu_2 = weighted_median(group2, w2)
            else:
                mu_1 = weighted_mean(group1, w1)
                mu_2 = weighted_mean(group2, w2)

            cov1 = weighted_cov(group1, w1)
            cov2 = weighted_cov(group2, w2)
            cov_pool = (cov1 + cov2) / 2
            cov_pool += np.eye(cov_pool.shape[0]) * 1e-6
            inv_cov = inv(cov_pool)
            return mahalanobis(mu_1, mu_2, inv_cov)
        return 0.0
    except (LinAlgError, ValueError):
        return 0.0


def compute_all_head_metrics_pca(X_heads, group_labels, group_values=(100, 200),
                                  n_components=10, centroid_method='mean', n_jobs=-1):
    N, L, H, D = X_heads.shape
    flat_heads = [X_heads[:, l, h, :] for l in range(L) for h in range(H)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mahalanobis_pca)(hd, group_labels, group_values, n_components, centroid_method)
        for hd in flat_heads
    )
    return np.array(results).reshape(L, H)


# =============================================================================
# CODE MAPS
# =============================================================================

def build_code_maps(all_fields, stata_2024_path, stata_2022_path):
    print('Loading value labels from Stata files...')
    df_num_24 = pd.read_stata(stata_2024_path, convert_categoricals=False)
    df_cat_24 = pd.read_stata(stata_2024_path, convert_categoricals=True)
    df_num_22 = pd.read_stata(stata_2022_path, convert_categoricals=False)
    reader_22 = pd.io.stata.StataReader(stata_2022_path)
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

    code_maps = {}
    for field in all_fields:
        code_maps[field] = _build_one(field)
    del df_num_24, df_cat_24, df_num_22
    gc.collect()
    return code_maps


# =============================================================================
# DEMOGRAPHIC PROFILES
# =============================================================================

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
            text = code_maps[field][code] if (field in code_maps and code in code_maps[field]) else str(code)
            parts.append((field, f'{label}: {text}'))
        all_parts[idx] = parts
    return all_parts


# =============================================================================
# STRATIFIED SAMPLING
# =============================================================================

def stratified_sample(df, frac, seed, min_per_group=1):
    rng = np.random.default_rng(seed)
    sampled = []
    for _, group in df.groupby('_strat_key'):
        n = max(min_per_group, int(np.ceil(len(group) * frac)))
        n = min(n, len(group))
        idx = rng.choice(group.index, size=n, replace=False)
        sampled.append(df.loc[idx])
    return pd.concat(sampled).sort_index()


# =============================================================================
# TOPIC ANALYSIS
# =============================================================================

def is_valid_response(val):
    if pd.isna(val):
        return False
    try:
        return int(float(val)) < 1000000
    except (ValueError, TypeError):
        return False


def run_topic_fast(model, tokenizer, topic_name, survey_question,
                   df_all, demo_parts, active_fields_set,
                   sample_frac, sample_seed,
                   prompt_fmt, system_msg,
                   pca_dims, batch_size, max_length):
    t0 = time.time()

    if topic_name not in df_all.columns:
        return None

    valid_mask = df_all[topic_name].apply(is_valid_response)
    df_valid = df_all[valid_mask]
    if len(df_valid) < 20:
        return None

    df_sampled = stratified_sample(df_valid, sample_frac, sample_seed)
    n_dem = (df_sampled['party_code'] == 100).sum()
    n_rep = (df_sampled['party_code'] == 200).sum()
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
        prompts.append(prompt_fmt.format(profile=profile, question=survey_question))

    labels = df_sampled['party_code'].values.astype(int)

    X_heads = extract_heads_batched(
        model, tokenizer, prompts, system_msg,
        batch_size=batch_size, max_length=max_length
    )

    results = {'Topic': topic_name, 'n_sampled': len(df_sampled),
               'n_dem': int(n_dem), 'n_rep': int(n_rep)}

    for centroid_method in ['mean', 'median']:
        suffix = '' if centroid_method == 'mean' else '_median'
        for n_comp in pca_dims:
            grid = compute_all_head_metrics_pca(
                X_heads, labels, group_values=(100, 200),
                n_components=n_comp, centroid_method=centroid_method
            )
            results[f'Avg_Mahal_PCA{n_comp}{suffix}'] = np.mean(grid)
            results[f'Max_Mahal_PCA{n_comp}{suffix}'] = np.max(grid)

    del X_heads

    elapsed = time.time() - t0
    d = pca_dims[0]
    excl = f' [excl {exclude_field}]' if exclude_field else ''
    print(f'    {topic_name}: mahal={results[f"Avg_Mahal_PCA{d}"]:.3f} '
          f'(n={len(df_sampled)}, D={n_dem}, R={n_rep}, {elapsed:.1f}s){excl}', flush=True)
    return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Standalone fast config testing (RTX 3090)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-id', type=str, required=True,
                        help='HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Short name for output files')
    parser.add_argument('--attn-impl', type=str, default='sdpa',
                        choices=['sdpa', 'flash_attention_2', 'eager'])
    parser.add_argument('--demo-fields', type=str, default='all',
                        choices=['core', 'expanded', 'all'])
    parser.add_argument('--prompt-fmt', type=str, default='B',
                        choices=['A', 'B', 'C'])
    parser.add_argument('--include-lifestyle', action='store_true')
    parser.add_argument('--sample-frac', type=float, default=0.01)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (8 is safe for 24GB 3090)')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--pca-dims', type=int, nargs='+', default=[15])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--min-dem', type=int, default=100)
    parser.add_argument('--min-rep', type=int, default=100)
    parser.add_argument('--min-total', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    prompt_fmt = PROMPT_FORMATS[args.prompt_fmt]

    # =========================================================================
    # BUILD ACTIVE FIELD LIST
    # =========================================================================
    print('=' * 70)
    print('STANDALONE FAST CONFIGURATION TESTING')
    print('=' * 70)

    demo_df = pd.read_csv(data_dir / 'gss_question_lists' / 'gss_demographic_variables.csv')
    demo_labels = dict(zip(demo_df['VariableName'], demo_df['ConciseDescription']))

    lifestyle_df = pd.read_csv(data_dir / 'gss_question_lists' / 'gss_politicized_lifestyle_variables.csv')
    lifestyle_labels = dict(zip(lifestyle_df['VariableName'], lifestyle_df['ConciseDescription']))

    if args.demo_fields == 'core':
        active_fields = [f for f in DEMO_FIELDS_CORE if f in DEMO_VARS_ALL_YEARS]
    elif args.demo_fields == 'expanded':
        active_fields = [f for f in DEMO_FIELDS_EXPANDED if f in DEMO_VARS_ALL_YEARS]
    else:
        active_fields = sorted(DEMO_VARS_ALL_YEARS)

    all_labels = {}
    all_labels.update(demo_labels)

    lifestyle_fields = set()
    if args.include_lifestyle:
        lifestyle_fields = LIFESTYLE_VARS_ALL_YEARS.copy()
        for var in sorted(lifestyle_fields):
            if var not in active_fields:
                active_fields.append(var)
        all_labels.update(lifestyle_labels)

    for f in active_fields:
        if f not in all_labels:
            all_labels[f] = f

    active_fields_set = set(active_fields)

    print(f'\n[Configuration]')
    print(f'  Model: {args.model_id} ({args.model_name})')
    print(f'  Demo fields: {args.demo_fields} ({len([f for f in active_fields if f not in lifestyle_fields])} fields)')
    print(f'  Lifestyle: {args.include_lifestyle} ({len([f for f in active_fields if f in lifestyle_fields])} fields)')
    print(f'  Total fields: {len(active_fields)}')
    print(f'  Prompt format: {args.prompt_fmt}')
    print(f'  Sample fraction: {args.sample_frac}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Max length: {args.max_length}')
    print(f'  Attention: {args.attn_impl}')
    print(f'  Data dir: {data_dir}')

    # =========================================================================
    # BUILD CODE MAPS
    # =========================================================================
    code_maps = build_code_maps(
        active_fields,
        str(data_dir / 'GSS2024.dta'),
        str(data_dir / 'GSS2022.dta'),
    )

    # =========================================================================
    # LOAD GSS DATA
    # =========================================================================
    print('\n' + '=' * 70)
    print('LOADING GSS DATA')
    print('=' * 70)

    df_gss = pd.read_csv(data_dir / 'gss_2021_2024.csv', low_memory=False)
    print(f'  Respondents: {df_gss.shape[0]}, Variables: {df_gss.shape[1]}')

    DEM_CODES = [0, 1, 2]
    REP_CODES = [4, 5, 6]
    df_gss['party_code'] = np.nan
    df_gss.loc[df_gss['partyid'].isin(DEM_CODES), 'party_code'] = 100
    df_gss.loc[df_gss['partyid'].isin(REP_CODES), 'party_code'] = 200
    df_dr = df_gss[df_gss['party_code'].notna()].copy()
    print(f'  Party coding:')
    print(f'    Democrat (100):   partyid in {DEM_CODES}')
    print(f'    Republican (200): partyid in {REP_CODES}')
    print(f'    Excluded:         partyid=3 (pure independent)')
    print(f'  D/R respondents: {len(df_dr)} '
          f'(D={int((df_dr["party_code"]==100).sum())}, R={int((df_dr["party_code"]==200).sum())})')

    partyid_labels = {0: 'Strong D', 1: 'Not strong D', 2: 'Ind near D',
                      3: 'Independent', 4: 'Ind near R', 5: 'Not strong R', 6: 'Strong R'}
    pid_counts = df_gss['partyid'].value_counts().sort_index()
    print(f'  Partyid distribution:')
    for pid in sorted(pid_counts.index):
        if pd.notna(pid) and int(pid) in partyid_labels:
            print(f'    {int(pid)} ({partyid_labels[int(pid)]}): {pid_counts[pid]}')

    df_dr['age_bin'] = pd.cut(df_dr['age'], bins=[0, 30, 40, 50, 60, 70, 100], labels=False)
    strat_cols_available = [c for c in STRATIFY_COLS if c in df_dr.columns]
    df_dr['_strat_key'] = ''
    for col in strat_cols_available:
        df_dr['_strat_key'] += df_dr[col].fillna(-1).astype(int).astype(str) + '_'
    print(f'  Stratification columns: {strat_cols_available}')
    print(f'  Stratification groups: {df_dr["_strat_key"].nunique()}')

    print('\nPre-computing demographic profiles...')
    demo_parts = precompute_demo_parts(df_dr, code_maps, active_fields, all_labels)
    print(f'  Done: {len(demo_parts)} respondents')

    # Show example prompt
    print(f'\n  Active fields ({len(active_fields)}): {active_fields}')
    ex_idx = list(demo_parts.keys())[0]
    ex_parts = [p[1] for p in demo_parts[ex_idx]]
    ex_profile = '. '.join(ex_parts) + '.'
    ex_prompt = prompt_fmt.format(profile=ex_profile, question='<SURVEY QUESTION>')
    print(f'\n  Example prompt (idx={ex_idx}):')
    print(f'  ---')
    for line in ex_prompt.split('\n'):
        print(f'  {line}')
    print(f'  ---')
    print(f'  Prompt length: {len(ex_prompt)} chars, ~{len(ex_prompt.split())} words')

    est_sample = stratified_sample(df_dr, args.sample_frac, args.sample_seed)
    print(f'\n  Estimated sample per topic: ~{len(est_sample)} (frac={args.sample_frac})')
    del est_sample

    # =========================================================================
    # LOAD TOPICS
    # =========================================================================
    print('\n' + '=' * 70)
    print('LOADING TOPICS')
    print('=' * 70)

    pub_topics = pd.read_csv(data_dir / 'gss_question_lists' / 'public_issues.csv')
    priv_topics = pd.read_csv(data_dir / 'gss_question_lists' / 'private_life.csv')
    pol_pub = pd.read_csv(data_dir / 'public_issues_polarization.csv')
    pol_priv = pd.read_csv(data_dir / 'private_life_polarization.csv')

    pol_pub = pol_pub[
        (pol_pub['n_dem'] >= args.min_dem) & (pol_pub['n_rep'] >= args.min_rep) &
        (pol_pub['n_total'] >= args.min_total)
    ]
    pol_priv = pol_priv[
        (pol_priv['n_dem'] >= args.min_dem) & (pol_priv['n_rep'] >= args.min_rep) &
        (pol_priv['n_total'] >= args.min_total)
    ]

    categories = {
        'public_issues': {
            'topics': {row['Variable']: row['SurveyQuestion']
                       for _, row in pub_topics.iterrows()
                       if row['Variable'] in set(pol_pub['variable'])},
            'polarization': pol_pub,
        },
        'private_life': {
            'topics': {row['Variable']: row['SurveyQuestion']
                       for _, row in priv_topics.iterrows()
                       if row['Variable'] in set(pol_priv['variable'])},
            'polarization': pol_priv,
        },
    }
    for cat, cfg in categories.items():
        print(f'  {cat}: {len(cfg["topics"])} topics')

    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    print('\n' + '=' * 70)
    print('LOADING MODEL')
    print('=' * 70)
    print(f'  CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

    model, tokenizer = load_model(args.model_id, attn_implementation=args.attn_impl)
    info = get_model_info(model)
    print(f'  Layers: {info["num_layers"]}, Heads: {info["num_heads"]}, Head dim: {info["head_dim"]}')

    # =========================================================================
    # RUN ANALYSIS
    # =========================================================================
    all_results = []

    for cat_name, cat_cfg in categories.items():
        print(f'\n{"="*70}')
        print(f'ANALYZING: {cat_name.upper()} ({len(cat_cfg["topics"])} topics)')
        print(f'{"="*70}')

        for i, (topic_name, survey_q) in enumerate(cat_cfg['topics'].items()):
            try:
                result = run_topic_fast(
                    model, tokenizer, topic_name, survey_q,
                    df_dr, demo_parts, active_fields_set,
                    sample_frac=args.sample_frac,
                    sample_seed=args.sample_seed,
                    prompt_fmt=prompt_fmt, system_msg=SYSTEM_MSG,
                    pca_dims=args.pca_dims,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
                if result is not None:
                    result['category'] = cat_name
                    all_results.append(result)

                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f'    ERROR on {topic_name}: {e}', flush=True)
                gc.collect()
                torch.cuda.empty_cache()

    df_results = pd.DataFrame(all_results)
    print(f'\nTotal topics analyzed: {len(df_results)}')

    # =========================================================================
    # CORRELATION
    # =========================================================================
    print('\n' + '=' * 70)
    print('CORRELATION WITH SURVEY POLARIZATION')
    print('=' * 70)

    pol_data = {'public_issues': pol_pub, 'private_life': pol_priv}
    summary_rows = []
    d = args.pca_dims[0]

    for cat_name in ['public_issues', 'private_life']:
        df_cat = df_results[df_results['category'] == cat_name]
        if len(df_cat) == 0:
            continue
        df_pol = pol_data[cat_name]
        df_merged = df_cat.merge(
            df_pol[['variable', 'polarization']].rename(
                columns={'variable': 'Topic', 'polarization': 'GSS_Polarization'}
            ), on='Topic', how='inner'
        )
        print(f'\n[{cat_name.upper()}] (n={len(df_merged)} topics)')
        for method in ['mean', 'median']:
            suffix = '' if method == 'mean' else '_median'
            col = f'Avg_Mahal_PCA{d}{suffix}'
            if col in df_merged.columns:
                r_p = df_merged[col].corr(df_merged['GSS_Polarization'])
                r_s = df_merged[col].corr(df_merged['GSS_Polarization'], method='spearman')
                print(f'  Mahal PCA-{d} ({method}): r={r_p:.4f}, rho={r_s:.4f}')
                summary_rows.append({
                    'model': args.model_name, 'n_fields': len(active_fields),
                    'demo_fields': args.demo_fields,
                    'lifestyle': 'yes' if args.include_lifestyle else 'no',
                    'prompt': args.prompt_fmt, 'sample': args.sample_frac,
                    'category': cat_name, 'centroid': method,
                    f'r_PCA{d}': round(r_p, 4), f'rho_PCA{d}': round(r_s, 4),
                    'n_topics': len(df_merged),
                })

    # =========================================================================
    # SAVE
    # =========================================================================
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    tag = f'{args.demo_fields}_{args.prompt_fmt}'
    if args.include_lifestyle:
        tag += '_lifestyle'

    summary_path = output_dir / f'test_config_{tag}_{args.model_name}_{timestamp}.csv'
    detail_path = output_dir / f'test_config_detail_{tag}_{args.model_name}_{timestamp}.pkl'
    df_summary.to_csv(summary_path, index=False)
    df_results.to_pickle(detail_path)
    print(f'\nSaved: {summary_path}')
    print(f'Saved: {detail_path}')
    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()
