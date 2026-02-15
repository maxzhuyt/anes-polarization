#!/usr/bin/env python3
"""
GSS Demographic Simulation — Activation Polarization

Instead of simulating politicians, this script feeds actual GSS respondent
demographics to the LLM and captures activation polarization on public/private
issue questions.

Usage:
    python run_demo_sim.py --model-path /path/to/model --model-name Llama-3.1-8B-Instruct
    python run_demo_sim.py --include-lifestyle  # include lifestyle variables in persona
    python run_demo_sim.py --help
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
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

from model_utils import load_model, extract_heads_batched, get_model_info
from run_gss_pca import compute_all_head_metrics_pca

# Defaults
DEFAULT_GSS_DATA = '../data/gss/gss_2021_2024.csv'
DEFAULT_STATA_2024 = '../data/gss/GSS2024.dta'
DEFAULT_STATA_2022 = '../data/gss/GSS2022.dta'
DEFAULT_DEMO_CSV = '../question_lists/gss_demographic_variables.csv'
DEFAULT_LIFESTYLE_CSV = '../question_lists/gss_politicized_lifestyle_variables.csv'

SYSTEM_MSG = 'You are simulating the views of an American.'


# =============================================================================
# CODE MAP BUILDING
# =============================================================================

def build_code_maps(all_fields, stata_2024_path, stata_2022_path):
    """Build code->label mappings from Stata files for all fields."""
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
        print(f'  {field}: {len(code_maps[field])} codes')

    del df_num_24, df_cat_24, df_num_22
    gc.collect()

    return code_maps


# =============================================================================
# DEMOGRAPHIC PROFILE BUILDING
# =============================================================================

def precompute_demo_parts(df, code_maps, all_fields, all_labels):
    """
    Pre-compute demographic/lifestyle key-value tuples for each respondent.
    Returns dict mapping row index → list of (field_name, 'Label: value') tuples.
    This allows per-topic filtering when a lifestyle variable overlaps with the survey question.
    """
    all_parts = {}
    for idx in df.index:
        row = df.loc[idx]
        parts = []
        for field in all_fields:
            val = row.get(field)
            if pd.isna(val):
                continue
            code = int(val)
            label = all_labels[field]
            if field in code_maps and code in code_maps[field]:
                text = code_maps[field][code]
            else:
                text = str(code)
            parts.append((field, f'{label}: {text}'))
        all_parts[idx] = parts
    return all_parts


# =============================================================================
# TOPIC ANALYSIS
# =============================================================================

def is_valid_response(val):
    """Check if a response code is valid."""
    if pd.isna(val):
        return False
    try:
        return int(float(val)) < 1000000
    except (ValueError, TypeError):
        return False


def run_topic(model, tokenizer, topic_name, survey_question, df_respondents,
              demo_parts, pca_dims, batch_size, max_length, lifestyle_fields):
    """
    Run activation extraction and PCA analysis for a single topic.

    If lifestyle_fields is provided and topic_name is in it, that field is
    excluded from each respondent's profile to avoid data leakage.
    """
    t0 = time.time()

    if topic_name not in df_respondents.columns:
        return None

    valid_mask = df_respondents[topic_name].apply(is_valid_response)
    df_valid = df_respondents[valid_mask]

    if len(df_valid) < 20:
        print(f'    {topic_name}: SKIP (only {len(df_valid)} valid respondents)', flush=True)
        return None

    n_dem = (df_valid['party_code'] == 100).sum()
    n_rep = (df_valid['party_code'] == 200).sum()
    if n_dem < 10 or n_rep < 10:
        print(f'    {topic_name}: SKIP (D={n_dem}, R={n_rep})', flush=True)
        return None

    # Check if this topic overlaps with lifestyle variables
    exclude_field = topic_name if (lifestyle_fields and topic_name in lifestyle_fields) else None

    # Build prompts using pre-computed demographic parts
    rng = np.random.default_rng(hash(topic_name) % (2**32))
    prompts = []
    for idx in df_valid.index:
        # Filter out the current topic if it's a lifestyle variable
        if exclude_field:
            parts = [p[1] for p in demo_parts[idx] if p[0] != exclude_field]
        else:
            parts = [p[1] for p in demo_parts[idx]]
        rng.shuffle(parts)
        profile = '. '.join(parts) + '.'
        prompts.append(f'{profile}\n\nSurvey question: {survey_question}')

    labels = df_valid['party_code'].values.astype(int)

    # Extract activations
    X_heads = extract_heads_batched(
        model, tokenizer, prompts, SYSTEM_MSG,
        batch_size=batch_size, max_length=max_length
    )

    # Compute PCA Mahalanobis for mean and median
    results = {'Topic': topic_name, 'n_valid': len(df_valid), 'n_dem': n_dem, 'n_rep': n_rep}

    for centroid_method in ['mean', 'median']:
        suffix = '' if centroid_method == 'mean' else '_median'
        for n_comp in pca_dims:
            grid = compute_all_head_metrics_pca(
                X_heads, labels,
                group_values=(100, 200),
                n_components=n_comp,
                centroid_method=centroid_method
            )
            results[f'Avg_Mahal_PCA{n_comp}{suffix}'] = np.mean(grid)
            results[f'Max_Mahal_PCA{n_comp}{suffix}'] = np.max(grid)

    del X_heads

    elapsed = time.time() - t0
    d = pca_dims[0]
    excluded_note = f' [excluded {exclude_field}]' if exclude_field else ''
    print(f'    {topic_name}: mahal={results[f"Avg_Mahal_PCA{d}"]:.3f} '
          f'(n={len(df_valid)}, D={n_dem}, R={n_rep}, {elapsed:.1f}s){excluded_note}', flush=True)

    return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='GSS Demographic Simulation — Activation Polarization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model (for output files)')
    parser.add_argument('--attn-impl', type=str, default='sdpa',
                        choices=['sdpa', 'flash_attention_2', 'eager'],
                        help='Attention implementation')

    # Data
    parser.add_argument('--gss-data', type=str, default=DEFAULT_GSS_DATA,
                        help='Path to GSS survey CSV')
    parser.add_argument('--stata-2024', type=str, default=DEFAULT_STATA_2024,
                        help='Path to GSS 2024 Stata file')
    parser.add_argument('--stata-2022', type=str, default=DEFAULT_STATA_2022,
                        help='Path to GSS 2022 Stata file')
    parser.add_argument('--demo-csv', type=str, default=DEFAULT_DEMO_CSV,
                        help='CSV with demographic variable definitions')
    parser.add_argument('--lifestyle-csv', type=str, default=DEFAULT_LIFESTYLE_CSV,
                        help='CSV with lifestyle variable definitions')
    parser.add_argument('--include-lifestyle', action='store_true',
                        help='Include politicized lifestyle variables in persona')
    parser.add_argument('--public-topics', type=str,
                        default='../question_lists/public_issues.csv',
                        help='CSV with public issues topics')
    parser.add_argument('--public-polarization', type=str,
                        default='../data/polarization/public_issues_polarization.csv',
                        help='CSV with public issues polarization data')
    parser.add_argument('--private-topics', type=str,
                        default='../question_lists/private_life.csv',
                        help='CSV with private life topics')
    parser.add_argument('--private-polarization', type=str,
                        default='../data/polarization/private_life_polarization.csv',
                        help='CSV with private life polarization data')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')

    # PCA
    parser.add_argument('--pca-dims', type=int, nargs='+', default=[15],
                        help='PCA dimensions to test')

    # Processing
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for model inference')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')

    # Filtering
    parser.add_argument('--min-dem', type=int, default=100,
                        help='Minimum Democrat respondents for GSS filtering')
    parser.add_argument('--min-rep', type=int, default=100,
                        help='Minimum Republican respondents for GSS filtering')
    parser.add_argument('--min-total', type=int, default=200,
                        help='Minimum total respondents for GSS filtering')

    # Exclusions
    parser.add_argument('--exclude-public', type=str, nargs='*',
                        default=['hubbywk1', 'racdif1', 'racdif2', 'racdif3', 'racdif4',
                                 'workwhts', 'wlthwhts', 'intlwhts'],
                        help='Topics to exclude from public issues')
    parser.add_argument('--exclude-private', type=str, nargs='*',
                        default=['reborn', 'marwht', 'helpful', 'helpfulnv', 'helpfulv'],
                        help='Topics to exclude from private life')

    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print('=' * 70)
    print('GSS DEMOGRAPHIC SIMULATION — ACTIVATION POLARIZATION')
    print('=' * 70)
    print(f'\n[Configuration]')
    print(f'  Model: {args.model_name}')
    print(f'  Model path: {args.model_path}')
    print(f'  Attention: {args.attn_impl}')
    print(f'  Include lifestyle: {args.include_lifestyle}')
    print(f'  PCA dimensions: {args.pca_dims}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Max length: {args.max_length}')
    print(f'  Output: {output_dir.absolute()}')
    print(f'  Timestamp: {timestamp}')

    # =========================================================================
    # LOAD DEMOGRAPHIC FIELDS
    # =========================================================================
    print('\n' + '=' * 70)
    print('LOADING DEMOGRAPHIC FIELDS')
    print('=' * 70)

    demo_df = pd.read_csv(args.demo_csv)
    all_fields = list(demo_df['VariableName'])
    all_labels = dict(zip(demo_df['VariableName'], demo_df['ConciseDescription']))

    if 'polviews' not in all_fields:
        all_fields.append('polviews')
        all_labels['polviews'] = 'Political views'

    print(f'  Demographic fields: {len(all_fields)}')

    # =========================================================================
    # LOAD LIFESTYLE FIELDS (optional)
    # =========================================================================
    lifestyle_fields = set()
    if args.include_lifestyle:
        print('\n' + '=' * 70)
        print('LOADING LIFESTYLE FIELDS')
        print('=' * 70)

        lifestyle_df = pd.read_csv(args.lifestyle_csv)
        lifestyle_fields = set(lifestyle_df['VariableName'])

        # Merge lifestyle into all_fields (avoid duplicates)
        for _, row in lifestyle_df.iterrows():
            var = row['VariableName']
            if var not in all_labels:
                all_fields.append(var)
                all_labels[var] = row['ConciseDescription']

        print(f'  Lifestyle fields: {len(lifestyle_fields)}')
        print(f'  Total fields (after merge): {len(all_fields)}')

    # =========================================================================
    # BUILD CODE MAPS
    # =========================================================================
    code_maps = build_code_maps(all_fields, args.stata_2024, args.stata_2022)

    # =========================================================================
    # LOAD GSS DATA
    # =========================================================================
    print('\n' + '=' * 70)
    print('LOADING GSS DATA')
    print('=' * 70)

    df_gss = pd.read_csv(args.gss_data, low_memory=False)
    print(f'  Respondents: {df_gss.shape[0]}, Variables: {df_gss.shape[1]}')
    print(f'  Years: {sorted(df_gss["year"].unique())}')

    # Filter to Democrats and Republicans
    # partyid: 0=Strong D, 1=Not strong D, 2=Ind near D,
    #          3=Independent, 4=Ind near R, 5=Not strong R, 6=Strong R
    df_gss['party_code'] = np.nan
    df_gss.loc[df_gss['partyid'].isin([0, 1, 2]), 'party_code'] = 100  # Democrat
    df_gss.loc[df_gss['partyid'].isin([4, 5, 6]), 'party_code'] = 200  # Republican
    df_dr = df_gss[df_gss['party_code'].notna()].copy()
    print(f'  D/R respondents: {len(df_dr)} '
          f'({(df_dr["party_code"]==100).sum()} D, {(df_dr["party_code"]==200).sum()} R)')

    # Pre-compute demographic parts
    print('\nPre-computing demographic profiles...')
    demo_parts = precompute_demo_parts(df_dr, code_maps, all_fields, all_labels)
    print(f'  Done: {len(demo_parts)} respondents')

    # =========================================================================
    # LOAD TOPICS
    # =========================================================================
    print('\n' + '=' * 70)
    print('LOADING TOPICS')
    print('=' * 70)

    pub_topics = pd.read_csv(args.public_topics)
    priv_topics = pd.read_csv(args.private_topics)

    pol_pub = pd.read_csv(args.public_polarization)
    pol_priv = pd.read_csv(args.private_polarization)

    pol_pub = pol_pub[
        (pol_pub['n_dem'] >= args.min_dem) &
        (pol_pub['n_rep'] >= args.min_rep) &
        (pol_pub['n_total'] >= args.min_total)
    ]
    pol_priv = pol_priv[
        (pol_priv['n_dem'] >= args.min_dem) &
        (pol_priv['n_rep'] >= args.min_rep) &
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

    # Check for overlap with lifestyle fields
    if lifestyle_fields:
        all_topic_vars = set()
        for cfg in categories.values():
            all_topic_vars.update(cfg['topics'].keys())
        overlap = lifestyle_fields & all_topic_vars
        if overlap:
            print(f'\n  NOTE: {len(overlap)} lifestyle fields overlap with survey topics')
            print(f'        These will be excluded from persona when asking that question:')
            print(f'        {sorted(overlap)}')

    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    print('\n' + '=' * 70)
    print('LOADING MODEL')
    print('=' * 70)
    print(f'  CUDA available: {torch.cuda.is_available()}')

    model, tokenizer = load_model(args.model_path, attn_implementation=args.attn_impl)
    model_info = get_model_info(model)
    print(f'  Layers: {model_info["num_layers"]}, Heads: {model_info["num_heads"]}, '
          f'Head dim: {model_info["head_dim"]}')

    # =========================================================================
    # RUN ANALYSIS
    # =========================================================================
    category_results = {}

    for cat_name, cat_cfg in categories.items():
        print(f'\n{"="*70}')
        print(f'ANALYZING: {cat_name.upper()}')
        print(f'{"="*70}')
        print(f'  Topics: {len(cat_cfg["topics"])}')
        print()

        results = []
        for idx, (topic_name, survey_q) in enumerate(cat_cfg['topics'].items()):
            try:
                result = run_topic(
                    model, tokenizer, topic_name, survey_q, df_dr,
                    demo_parts,
                    pca_dims=args.pca_dims, batch_size=args.batch_size,
                    max_length=args.max_length,
                    lifestyle_fields=lifestyle_fields
                )
                if result is not None:
                    result['category'] = cat_name
                    results.append(result)

                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f'    ERROR on {topic_name}: {e}', flush=True)
                gc.collect()
                torch.cuda.empty_cache()

        df_result = pd.DataFrame(results)
        category_results[cat_name] = df_result

        # Include lifestyle suffix in output filename
        lifestyle_suffix = '_lifestyle' if args.include_lifestyle else ''
        out_path = output_dir / f'df_demo_sim_{cat_name}{lifestyle_suffix}_{args.model_name}_{timestamp}.pkl'
        df_result.to_pickle(out_path)
        print(f'  Saved: {out_path}')

    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    print('\n' + '=' * 70)
    print('CORRELATION ANALYSIS')
    print('=' * 70)

    all_correlations = []

    for cat_name, cat_cfg in categories.items():
        df_llm = category_results[cat_name]
        df_pol = cat_cfg['polarization']

        df_merged = df_llm.merge(
            df_pol[['variable', 'polarization', 'area']].rename(
                columns={'variable': 'Topic', 'polarization': 'GSS_Polarization'}
            ),
            on='Topic', how='inner'
        )

        print(f'\n[{cat_name.upper()}]')
        for method in ['mean', 'median']:
            suffix = '' if method == 'mean' else '_median'
            print(f'  Centroid: {method}')
            for d in args.pca_dims:
                col = f'Avg_Mahal_PCA{d}{suffix}'
                if col in df_merged.columns:
                    r_p = df_merged[col].corr(df_merged['GSS_Polarization'], method='pearson')
                    r_s = df_merged[col].corr(df_merged['GSS_Polarization'], method='spearman')
                    print(f'    PCA-{d:2d}: r={r_p:.4f}, rho={r_s:.4f} (n={len(df_merged)})')
                    all_correlations.append({
                        'category': cat_name,
                        'pca_dim': d,
                        'centroid_method': method,
                        'n_topics': len(df_merged),
                        'pearson': r_p,
                        'spearman': r_s,
                        'df_merged': df_merged,
                        'llm_col': col,
                    })

    lifestyle_suffix = '_lifestyle' if args.include_lifestyle else ''
    df_corr = pd.DataFrame([{k: v for k, v in c.items() if k != 'df_merged'}
                             for c in all_correlations])
    corr_path = output_dir / f'demo_sim_correlations{lifestyle_suffix}_{args.model_name}_{timestamp}.csv'
    df_corr.to_csv(corr_path, index=False)
    print(f'\nSaved: {corr_path}')

    # =========================================================================
    # FILTERED RESULTS
    # =========================================================================
    print('\n' + '=' * 70)
    print('FILTERED RESULTS (Topic Exclusion)')
    print('=' * 70)

    excluded_sets = {
        'public_issues': set(args.exclude_public) if args.exclude_public else set(),
        'private_life': set(args.exclude_private) if args.exclude_private else set(),
    }

    filtered_results = []

    for corr in all_correlations:
        cat_name = corr['category']
        df_merged = corr['df_merged'].copy()
        llm_col = corr['llm_col']

        excluded = excluded_sets[cat_name]
        df_filtered = df_merged[~df_merged['Topic'].isin(excluded)].reset_index(drop=True)

        r_p = df_filtered[llm_col].corr(df_filtered['GSS_Polarization'], method='pearson')
        r_s = df_filtered[llm_col].corr(df_filtered['GSS_Polarization'], method='spearman')

        filtered_results.append({
            'Category': cat_name,
            'PCA_Dim': corr['pca_dim'],
            'Centroid_Method': corr['centroid_method'],
            'N_Original': len(df_merged),
            'N_Filtered': len(df_filtered),
            'Pearson_Original': corr['pearson'],
            'Pearson_Filtered': r_p,
            'Spearman_Original': corr['spearman'],
            'Spearman_Filtered': r_s,
        })

    df_filt = pd.DataFrame(filtered_results)
    print(f'\nExcluded (public): {sorted(excluded_sets["public_issues"])}')
    print(f'Excluded (private): {sorted(excluded_sets["private_life"])}')
    print(f'\n{df_filt.to_string()}')

    filt_path = output_dir / f'demo_sim_filtered{lifestyle_suffix}_{args.model_name}_{timestamp}.csv'
    df_filt.to_csv(filt_path, index=False)
    print(f'\nSaved: {filt_path}')

    # =========================================================================
    # COMBINED OUTPUT
    # =========================================================================
    df_combined = pd.concat(list(category_results.values()), ignore_index=True)
    df_combined['model'] = args.model_name
    df_combined['include_lifestyle'] = args.include_lifestyle

    combined_path = output_dir / f'df_demo_sim_combined{lifestyle_suffix}_{args.model_name}_{timestamp}'
    df_combined.to_csv(f'{combined_path}.csv', index=False)
    df_combined.to_pickle(f'{combined_path}.pkl')

    print(f'\nSaved: {combined_path}.csv')
    print(f'Total topics analyzed: {len(df_combined)}')

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'\nModel: {args.model_name}')
    print(f'Include lifestyle: {args.include_lifestyle}')
    print(f'PCA Dimensions: {args.pca_dims}')
    print(f'Attention: {args.attn_impl}')
    print(f'\nFiles saved to: {output_dir.absolute()}')
    print(f'  - {corr_path.name}')
    print(f'  - {filt_path.name}')
    print(f'  - {combined_path.name}.csv / .pkl')

    print('\n' + '=' * 70)
    print('ANALYSIS COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
