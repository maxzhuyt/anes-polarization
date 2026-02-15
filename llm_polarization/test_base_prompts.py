#!/usr/bin/env python3
"""
Base Model Prompt Sensitivity Test

Tests whether base models are sensitive to prompt formatting by comparing
two different completion-style prompt formats:
  - Original: "{name} shares views on {topic}, saying"
  - Alternative: "{name} makes statement on {topic}:"

Runs on a subset of 20 topics per category across all base models.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import pandas as pd
import numpy as np
import torch
import gc
import time
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')

from model_utils import load_model, extract_heads_batched, get_model_info
from prompt_utils import load_politicians
from run_gss_pca import compute_all_head_metrics_pca, load_topics_from_csv, load_polarization_data

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = "/project/jevans/maxzhuyt/models"

BASE_MODELS = {
    "SmolLM3-3B": {"path": f"{MODELS_DIR}/SmolLM3-3B-Base", "batch_size": 200},
    "Qwen3-4B": {"path": f"{MODELS_DIR}/Qwen3-4B-Base", "batch_size": 180},
    "Qwen2.5-7B": {"path": f"{MODELS_DIR}/Qwen2.5-7B", "batch_size": 160},
    "Llama-3.1-8B": {"path": f"{MODELS_DIR}/Meta-Llama-3.1-8B", "batch_size": 150},
    "Gemma-2-9b": {"path": f"{MODELS_DIR}/gemma-2-9b", "batch_size": 120},
    # Ministral-3-8B skipped: multimodal model, not compatible with text-only analysis
}

# Original prompts (current in run_model_comparison.py)
ORIGINAL_PROMPTS = {
    "public_issues": "{name} shares views on {topic}, saying",
    "private_life": "When asked about {topic}, {name} says",
}

# Alternative prompts (testing)
ALTERNATIVE_PROMPTS = {
    "public_issues": "{name} makes statement on {topic}:",
    "private_life": "On {topic}, {name} would say:",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_base_prompts(topic_desc, politician_names, template):
    """Generate completion-style prompts for base models."""
    return [template.format(name=name, topic=topic_desc) for name in politician_names]


def run_topic(model, tokenizer, topic_name, topic_desc, category,
              politician_names, politician_labels, prompt_template,
              pca_dims, batch_size, max_length):
    """Run extraction + PCA Mahalanobis for one topic with given prompt."""
    t0 = time.time()

    prompts = generate_base_prompts(topic_desc, politician_names, prompt_template)

    X_heads = extract_heads_batched(
        model, tokenizer, prompts, system_msg="",
        batch_size=batch_size, max_length=max_length,
    )

    results = {'Topic': topic_name, 'n_politicians': len(prompts)}

    for centroid_method in ['mean']:
        for n_comp in pca_dims:
            grid = compute_all_head_metrics_pca(
                X_heads, politician_labels,
                group_values=(100, 200),
                n_components=n_comp,
                centroid_method=centroid_method,
            )
            results[f'Avg_Mahal_PCA{n_comp}'] = float(np.mean(grid))
            results[f'Max_Mahal_PCA{n_comp}'] = float(np.max(grid))

    del X_heads
    elapsed = time.time() - t0
    print(f"    {topic_name}: {results['Avg_Mahal_PCA15']:.3f} ({elapsed:.1f}s)", flush=True)
    return results


def test_single_model(model_name, model_path, all_topics, politician_names,
                      politician_labels, pca_dims, batch_size, max_length):
    """Test one base model with both prompt formats."""
    print(f'\n{"="*70}')
    print(f'MODEL: {model_name}')
    print(f'  Path: {model_path}')
    print(f'  Batch size: {batch_size}')
    print(f'{"="*70}', flush=True)

    model, tokenizer = load_model(model_path)
    model_info = get_model_info(model)
    print(f'  Layers={model_info["num_layers"]}, Heads={model_info["num_heads"]}, '
          f'Head_dim={model_info["head_dim"]}')

    # Disable chat template for base model
    tokenizer.chat_template = None
    print('  [Base model] Disabled chat template for completion-style prompts\n')

    results = {'original': [], 'alternative': []}

    for prompt_type, prompts_dict in [('original', ORIGINAL_PROMPTS),
                                       ('alternative', ALTERNATIVE_PROMPTS)]:
        print(f'  --- TESTING {prompt_type.upper()} PROMPTS ---', flush=True)

        for cat_name in ["public_issues", "private_life"]:
            topics = all_topics[cat_name]
            template = prompts_dict[cat_name]

            print(f'\n  {cat_name.upper()} ({len(topics)} topics)')
            print(f'  Template: "{template}"', flush=True)

            for topic_name, topic_desc in topics.items():
                try:
                    result = run_topic(
                        model, tokenizer, topic_name, topic_desc, cat_name,
                        politician_names, politician_labels, template,
                        pca_dims, batch_size, max_length,
                    )
                    result['category'] = cat_name
                    result['model_name'] = model_name
                    result['prompt_type'] = prompt_type
                    results[prompt_type].append(result)
                except Exception as e:
                    print(f'    ERROR on {topic_name}: {e}', flush=True)

                gc.collect()
                torch.cuda.empty_cache()

    # Unload model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f'\n  Model unloaded.', flush=True)

    return results


def compute_correlations(df, pol_data):
    """Compute correlations with GSS polarization."""
    rows = []
    for cat_name in ['public_issues', 'private_life']:
        df_cat = df[df['category'] == cat_name]
        pol_df = pol_data[cat_name]

        df_merged = df_cat.merge(
            pol_df[['variable', 'polarization']].rename(
                columns={'variable': 'Topic', 'polarization': 'GSS_Polarization'}
            ),
            on='Topic', how='inner'
        )

        if len(df_merged) < 5:
            continue

        col = 'Avg_Mahal_PCA15'
        r, r_p = pearsonr(df_merged[col], df_merged['GSS_Polarization'])
        rho, rho_p = spearmanr(df_merged[col], df_merged['GSS_Polarization'])

        rows.append({
            'category': cat_name,
            'n_topics': len(df_merged),
            'pearson_r': round(r, 4),
            'pearson_p': round(r_p, 4),
            'spearman_rho': round(rho, 4),
            'spearman_p': round(rho_p, 4),
        })

    return pd.DataFrame(rows)


def compare_prompt_sets(df_orig, df_alt, model_name):
    """Compare polarization scores between original and alternative prompts."""
    print(f'\n{"="*70}')
    print(f'PROMPT COMPARISON: {model_name}')
    print(f'{"="*70}')

    # Merge on topic + category
    df_merged = df_orig.merge(
        df_alt, on=['Topic', 'category'], suffixes=('_orig', '_alt')
    )

    print(f'\nMatched topics: {len(df_merged)}')
    print(f'  Public issues: {len(df_merged[df_merged["category"] == "public_issues"])}')
    print(f'  Private life: {len(df_merged[df_merged["category"] == "private_life"])}')

    # Correlation between the two prompt sets
    col_orig = 'Avg_Mahal_PCA15_orig'
    col_alt = 'Avg_Mahal_PCA15_alt'

    r_overall, _ = pearsonr(df_merged[col_orig], df_merged[col_alt])
    rho_overall, _ = spearmanr(df_merged[col_orig], df_merged[col_alt])

    print(f'\n{"─"*70}')
    print('CORRELATION BETWEEN ORIGINAL AND ALTERNATIVE PROMPTS')
    print(f'{"─"*70}')
    print(f'  Overall (all topics):')
    print(f'    Pearson r:   {r_overall:.4f}')
    print(f'    Spearman rho: {rho_overall:.4f}')

    # Per-category
    for cat in ['public_issues', 'private_life']:
        df_cat = df_merged[df_merged['category'] == cat]
        if len(df_cat) < 5:
            continue
        r, _ = pearsonr(df_cat[col_orig], df_cat[col_alt])
        rho, _ = spearmanr(df_cat[col_orig], df_cat[col_alt])
        print(f'\n  {cat}:')
        print(f'    Pearson r:    {r:.4f}')
        print(f'    Spearman rho: {rho:.4f}')

    return df_merged


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test base model prompt sensitivity'
    )
    parser.add_argument('--politician-csv', type=str,
                        default='/project/jevans/maxzhuyt/data/HS116_members_fullname.csv')
    parser.add_argument('--public-topics', type=str,
                        default='../question_lists/public_issues.csv')
    parser.add_argument('--public-polarization', type=str,
                        default='../data/polarization/public_issues_polarization.csv')
    parser.add_argument('--private-topics', type=str,
                        default='../question_lists/private_life.csv')
    parser.add_argument('--private-polarization', type=str,
                        default='../data/polarization/private_life_polarization.csv')
    parser.add_argument('--n-topics', type=int, default=20,
                        help='Number of topics to sample per category')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--pca-dims', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)

    print('=' * 70)
    print('BASE MODEL PROMPT SENSITIVITY TEST')
    print('=' * 70)
    print(f'  Timestamp: {timestamp}')
    print(f'  Base models: {len(BASE_MODELS)}')
    print(f'  Topics per category: {args.n_topics}')
    print(f'  PCA dims: {args.pca_dims}')
    print(f'  Output dir: {output_dir}')
    print('=' * 70, flush=True)

    # Load politicians
    df_politicians = load_politicians(args.politician_csv)
    df_politicians = df_politicians[df_politicians['party_code'].isin([100, 200])].dropna(subset=['fullname'])
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = df_politicians['party_code'].values
    n_dem = int((politician_labels == 100).sum())
    n_rep = int((politician_labels == 200).sum())
    print(f'\nPoliticians: {len(politician_names)} (D={n_dem}, R={n_rep})')

    # Load topics
    pub_topics_full, _ = load_topics_from_csv(args.public_topics)
    priv_topics_full, _ = load_topics_from_csv(args.private_topics)

    # Load polarization data for filtering
    pol_pub = load_polarization_data(args.public_polarization)
    pol_priv = load_polarization_data(args.private_polarization)

    pol_pub = pol_pub[(pol_pub['n_dem'] >= 100) & (pol_pub['n_rep'] >= 100) & (pol_pub['n_total'] >= 200)]
    pol_priv = pol_priv[(pol_priv['n_dem'] >= 100) & (pol_priv['n_rep'] >= 100) & (pol_priv['n_total'] >= 200)]

    pub_valid = set(pol_pub['variable'])
    priv_valid = set(pol_priv['variable'])
    pub_topics_full = {k: v for k, v in pub_topics_full.items() if k in pub_valid}
    priv_topics_full = {k: v for k, v in priv_topics_full.items() if k in priv_valid}

    # Sample topics
    pub_topics_list = list(pub_topics_full.items())
    priv_topics_list = list(priv_topics_full.items())

    pub_sampled = dict(pub_topics_list[:args.n_topics])
    priv_sampled = dict(priv_topics_list[:args.n_topics])

    all_topics = {
        'public_issues': pub_sampled,
        'private_life': priv_sampled,
    }

    print(f'Public issues topics: {len(pub_sampled)} (sampled from {len(pub_topics_full)})')
    print(f'Private life topics: {len(priv_sampled)} (sampled from {len(priv_topics_full)})')
    print()

    pol_data = {
        'public_issues': pol_pub,
        'private_life': pol_priv,
    }

    # Run tests
    all_results = []
    comparison_results = []

    for model_name, model_cfg in BASE_MODELS.items():
        try:
            results = test_single_model(
                model_name=model_name,
                model_path=model_cfg['path'],
                all_topics=all_topics,
                politician_names=politician_names,
                politician_labels=politician_labels,
                pca_dims=args.pca_dims,
                batch_size=model_cfg['batch_size'],
                max_length=args.max_length,
            )

            df_orig = pd.DataFrame(results['original'])
            df_alt = pd.DataFrame(results['alternative'])

            # Compare prompts
            df_comparison = compare_prompt_sets(df_orig, df_alt, model_name)

            # Compute GSS correlations
            print(f'\n{"─"*70}')
            print('CORRELATION WITH GSS POLARIZATION')
            print(f'{"─"*70}')

            corr_orig = compute_correlations(df_orig, pol_data)
            corr_alt = compute_correlations(df_alt, pol_data)

            print('\nORIGINAL PROMPTS:')
            print(corr_orig.to_string(index=False))

            print('\nALTERNATIVE PROMPTS:')
            print(corr_alt.to_string(index=False))

            # Compare correlations
            print(f'\n{"─"*70}')
            print('COMPARISON TABLE')
            print(f'{"─"*70}')
            comp_table = corr_orig.merge(corr_alt, on='category', suffixes=('_orig', '_alt'))
            comp_table['pearson_diff'] = comp_table['pearson_r_alt'] - comp_table['pearson_r_orig']
            comp_table['spearman_diff'] = comp_table['spearman_rho_alt'] - comp_table['spearman_rho_orig']

            print(comp_table[['category', 'pearson_r_orig', 'pearson_r_alt', 'pearson_diff',
                             'spearman_rho_orig', 'spearman_rho_alt', 'spearman_diff']].to_string(index=False))

            # Store for aggregate analysis
            comp_table['model_name'] = model_name
            comparison_results.append(comp_table)

            all_results.append({
                'model_name': model_name,
                'df_original': df_orig,
                'df_alternative': df_alt,
                'corr_original': corr_orig,
                'corr_alternative': corr_alt,
            })

        except Exception as e:
            print(f'\nERROR on {model_name}: {e}', flush=True)
            import traceback
            traceback.print_exc()

    # Aggregate summary
    print(f'\n{"="*70}')
    print('AGGREGATE SUMMARY ACROSS ALL MODELS')
    print(f'{"="*70}')

    if comparison_results:
        df_all_comp = pd.concat(comparison_results, ignore_index=True)

        print('\nPer-model, per-category correlation differences:')
        print(df_all_comp[['model_name', 'category', 'pearson_r_orig', 'pearson_r_alt',
                          'pearson_diff', 'spearman_diff']].to_string(index=False))

        print(f'\n{"─"*70}')
        print('AVERAGE DIFFERENCES ACROSS MODELS:')
        print(f'{"─"*70}')
        for cat in ['public_issues', 'private_life']:
            df_cat = df_all_comp[df_all_comp['category'] == cat]
            if len(df_cat) == 0:
                continue
            print(f'\n{cat}:')
            print(f'  Mean Pearson difference:   {df_cat["pearson_diff"].mean():+.4f}')
            print(f'  Mean Spearman difference:  {df_cat["spearman_diff"].mean():+.4f}')
            print(f'  Std Pearson difference:    {df_cat["pearson_diff"].std():.4f}')
            print(f'  Std Spearman difference:   {df_cat["spearman_diff"].std():.4f}')

        # Save results
        summary_path = output_dir / f'base_prompt_test_summary_{timestamp}.csv'
        df_all_comp.to_csv(summary_path, index=False)
        print(f'\n  >> Saved summary: {summary_path}')

    print(f'\n{"="*70}')
    print('DONE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
