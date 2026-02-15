#!/usr/bin/env python3
"""
Model Comparison: Base vs Instruct vs Reasoning Polarization Alignment

Compares base, instruct, and reasoning models within the same model family
to measure how fine-tuning affects alignment with survey-measured political
polarization.

Usage:
    python run_model_comparison.py
    python run_model_comparison.py --families Qwen3-4B Llama-3.1-8B
    python run_model_comparison.py --batch-size 64 --pca-dims 10 15
    sbatch run_model_comparison.sbatch
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
import warnings
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from config import SYSTEM_MSG_POLITICIAN
from model_utils import load_model, extract_heads_batched, get_model_info
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from run_gss_pca import compute_all_head_metrics_pca, load_topics_from_csv, load_polarization_data

# =============================================================================
# MODEL FAMILY CONFIGURATION
# =============================================================================

MODELS_DIR = "/project/jevans/maxzhuyt/models"

MODEL_FAMILIES = {
    "Gemma-2-9b": {
        "base":      {"path": f"{MODELS_DIR}/gemma-2-9b",    "type": "base",     "batch_size": 160},
        "instruct":  {"path": f"{MODELS_DIR}/gemma-2-9b-it", "type": "instruct", "batch_size": 160},
    },

    "SmolLM3-3B": {
        "base":      {"path": f"{MODELS_DIR}/SmolLM3-3B-Base",            "type": "base",      "batch_size": 138},
        "instruct":  {"path": f"{MODELS_DIR}/SmolLM3-3B",                 "type": "instruct",  "batch_size": 138,
                      "system_override": "/no_think"},
        "reasoning": {"path": f"{MODELS_DIR}/SmolLM3-3B",                 "type": "reasoning", "batch_size": 138,
                      "system_override": "/think"},
    },
    "Qwen3-4B": {
        "base":      {"path": f"{MODELS_DIR}/Qwen3-4B-Base",              "type": "base",      "batch_size": 128},
        "instruct":  {"path": f"{MODELS_DIR}/Qwen3-4B-Instruct-2507",     "type": "instruct",  "batch_size": 128},
        "reasoning": {"path": f"{MODELS_DIR}/Qwen3-4B-Thinking-2507",     "type": "reasoning", "batch_size": 128},
    },
    "Qwen2.5-7B": {
        "base":      {"path": f"{MODELS_DIR}/Qwen2.5-7B",                 "type": "base",      "batch_size": 128},
        "instruct":  {"path": f"{MODELS_DIR}/Qwen2.5-7B-Instruct",        "type": "instruct",  "batch_size": 128},
    },
    "Llama-3.1-8B": {
        "base":      {"path": f"{MODELS_DIR}/Meta-Llama-3.1-8B",          "type": "base",      "batch_size": 96},
        "instruct":  {"path": f"{MODELS_DIR}/Meta-Llama-3.1-8B-Instruct", "type": "instruct",  "batch_size": 96},
        "reasoning": {"path": f"{MODELS_DIR}/DeepSeek-R1-Distill-Llama-8B", "type": "reasoning", "batch_size": 96},
    },
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# Completion-style prompts for base models (no chat template)
BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}

# Which POLITICIAN_TEMPLATES key to use for instruct/reasoning models
INSTRUCT_TEMPLATE_KEYS = {
    "public_issues": "default",   # "Generate a statement by {name} on {topic}."
    "private_life":  "opinion",   # "What would {name} say about {topic}?"
}


def get_prompts_and_system_msg(topic_desc, politician_names, model_type, category,
                               system_override=None):
    """Generate prompts and system message appropriate for the model type.

    Args:
        system_override: If set, replaces the default system message for
            instruct/reasoning models (e.g., "/no_think" or "/think" for SmolLM3).
    """
    if model_type == "base":
        template = BASE_TEMPLATES[category]
        prompts = [template.format(name=name, topic=topic_desc) for name in politician_names]
        system_msg = ""
    else:
        template_key = INSTRUCT_TEMPLATE_KEYS[category]
        template = POLITICIAN_TEMPLATES[template_key]
        prompts = generate_politician_prompts(topic_desc, politician_names, template=template)
        system_msg = system_override if system_override is not None else SYSTEM_MSG_POLITICIAN
    return prompts, system_msg


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def run_topic_for_comparison(model, tokenizer, topic_name, topic_desc,
                             model_type, category, politician_names,
                             politician_labels, pca_dims, batch_size, max_length,
                             system_override=None):
    """Run extraction + PCA Mahalanobis for one topic on one model."""
    t0 = time.time()

    prompts, system_msg = get_prompts_and_system_msg(
        topic_desc, politician_names, model_type, category,
        system_override=system_override,
    )

    X_heads = extract_heads_batched(
        model, tokenizer, prompts, system_msg,
        batch_size=batch_size, max_length=max_length,
    )

    results = {'Topic': topic_name, 'n_politicians': len(prompts)}

    for centroid_method in ['mean', 'median']:
        suffix = '' if centroid_method == 'mean' else '_median'
        for n_comp in pca_dims:
            grid = compute_all_head_metrics_pca(
                X_heads, politician_labels,
                group_values=(100, 200),
                n_components=n_comp,
                centroid_method=centroid_method,
            )
            results[f'Avg_Mahal_PCA{n_comp}{suffix}'] = float(np.mean(grid))
            results[f'Max_Mahal_PCA{n_comp}{suffix}'] = float(np.max(grid))

    del X_heads
    elapsed = time.time() - t0
    d = pca_dims[0]
    print(f"    {topic_name}: avg={results[f'Avg_Mahal_PCA{d}']:.3f} ({elapsed:.1f}s)", flush=True)
    return results


def run_single_model(model_path, model_name, model_type, family_name,
                     all_topics, politician_names, politician_labels,
                     pca_dims, batch_size, max_length, output_dir, timestamp,
                     system_override=None):
    """Load one model, run all topics, save checkpoint, unload."""
    print(f'\n{"="*70}')
    print(f'MODEL: {model_name} (family={family_name}, type={model_type})')
    print(f'  Path: {model_path}')
    print(f'  Batch size: {batch_size}')
    if system_override is not None:
        print(f'  System override: "{system_override}"')
    print(f'{"="*70}', flush=True)

    model, tokenizer = load_model(model_path)
    model_info = get_model_info(model)
    print(f'  Layers={model_info["num_layers"]}, Heads={model_info["num_heads"]}, '
          f'Head_dim={model_info["head_dim"]}')

    # For base models, disable chat template to force completion-style prompts
    if model_type == "base":
        tokenizer.chat_template = None
        print('  [Base model] Disabled chat template for completion-style prompts')

    # Show example prompt
    sample_prompts, sample_sys = get_prompts_and_system_msg(
        "immigration policy", [politician_names[0]], model_type, "public_issues",
        system_override=system_override,
    )
    print(f'  Example prompt: "{sample_prompts[0]}"')
    if sample_sys:
        print(f'  System msg: "{sample_sys[:80]}..."')
    print(flush=True)

    results = []
    for cat_name in ["public_issues", "private_life"]:
        topics = all_topics[cat_name]
        print(f'\n  --- {cat_name.upper()} ({len(topics)} topics) ---', flush=True)

        for topic_name, topic_desc in topics.items():
            try:
                result = run_topic_for_comparison(
                    model, tokenizer, topic_name, topic_desc,
                    model_type, cat_name, politician_names, politician_labels,
                    pca_dims, batch_size, max_length,
                    system_override=system_override,
                )
                result['category'] = cat_name
                result['model_name'] = model_name
                result['model_type'] = model_type
                result['family'] = family_name
                results.append(result)
            except Exception as e:
                print(f'    ERROR on {topic_name}: {e}', flush=True)

            gc.collect()
            torch.cuda.empty_cache()

    # Unload model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f'\n  Model unloaded. {len(results)} topics completed.', flush=True)

    # Save per-model checkpoint
    df_model = pd.DataFrame(results)
    safe_name = model_name.replace('/', '_')
    ckpt_path = output_dir / f'comparison_{safe_name}_{timestamp}.pkl'
    df_model.to_pickle(ckpt_path)
    print(f'  >> Saved: {ckpt_path}', flush=True)

    return df_model


# =============================================================================
# CORRELATION + FILTERING
# =============================================================================

def compute_comparison_correlations(df_all, pol_data, pca_dims, exclude_pub=None, exclude_priv=None):
    """Compute Pearson/Spearman correlations for each model × category."""
    exclude_map = {
        'public_issues': set(exclude_pub or []),
        'private_life': set(exclude_priv or []),
    }

    rows = []
    for (model_name, model_type, family), df_model in df_all.groupby(
        ['model_name', 'model_type', 'family']
    ):
        for cat_name in ['public_issues', 'private_life']:
            df_cat = df_model[df_model['category'] == cat_name]
            if len(df_cat) == 0:
                continue

            excl = exclude_map[cat_name]
            if excl:
                df_cat = df_cat[~df_cat['Topic'].isin(excl)]

            df_pol = pol_data[cat_name]
            if excl:
                df_pol = df_pol[~df_pol['variable'].isin(excl)]

            df_merged = df_cat.merge(
                df_pol[['variable', 'polarization']].rename(
                    columns={'variable': 'Topic', 'polarization': 'GSS_Polarization'}
                ),
                on='Topic', how='inner'
            )
            if len(df_merged) < 5:
                continue

            for centroid in ['mean', 'median']:
                suffix = '' if centroid == 'mean' else '_median'
                for d in pca_dims:
                    col = f'Avg_Mahal_PCA{d}{suffix}'
                    if col not in df_merged.columns:
                        continue
                    r, r_p = pearsonr(df_merged[col], df_merged['GSS_Polarization'])
                    rho, rho_p = spearmanr(df_merged[col], df_merged['GSS_Polarization'])
                    rows.append({
                        'family': family,
                        'model_name': model_name,
                        'model_type': model_type,
                        'category': cat_name,
                        'centroid': centroid,
                        'pca_dim': d,
                        'n_topics': len(df_merged),
                        'pearson_r': round(r, 4),
                        'pearson_p': round(r_p, 4),
                        'spearman_rho': round(rho, 4),
                        'spearman_p': round(rho_p, 4),
                    })

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison_bar_chart(df_corr, pca_dim, centroid, output_path):
    """Grouped bar chart: correlation by model type per family."""
    suffix = '' if centroid == 'mean' else '_median'
    df_plot = df_corr[
        (df_corr['pca_dim'] == pca_dim) & (df_corr['centroid'] == centroid)
    ].copy()

    if len(df_plot) == 0:
        print('  No data for bar chart', flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    type_colors = {'base': '#4C72B0', 'instruct': '#DD8452', 'reasoning': '#55A868'}
    type_order = ['base', 'instruct', 'reasoning']

    for ax, cat_name in zip(axes, ['public_issues', 'private_life']):
        df_cat = df_plot[df_plot['category'] == cat_name]
        families = df_cat['family'].unique()
        n_families = len(families)
        bar_width = 0.25
        x = np.arange(n_families)

        for j, mtype in enumerate(type_order):
            vals = []
            for fam in families:
                row = df_cat[(df_cat['family'] == fam) & (df_cat['model_type'] == mtype)]
                vals.append(row['pearson_r'].values[0] if len(row) > 0 else 0)
            ax.bar(x + j * bar_width, vals, bar_width,
                   label=mtype, color=type_colors[mtype], alpha=0.85)

        ax.set_xlabel('Model Family')
        ax.set_ylabel('Pearson r')
        ax.set_title(f'{cat_name.replace("_", " ").title()}')
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(families, rotation=15, ha='right')
        ax.legend()
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    fig.suptitle(f'LLM Polarization vs Survey Polarization (PCA-{pca_dim}, {centroid} centroid)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  >> Saved bar chart: {output_path}', flush=True)


def plot_scatter_grid(df_all, pol_data, pca_dim, centroid, output_path):
    """Scatter grid: one row per model, columns = categories."""
    suffix = '' if centroid == 'mean' else '_median'
    col = f'Avg_Mahal_PCA{pca_dim}{suffix}'

    models = df_all.groupby(['family', 'model_type', 'model_name']).size().reset_index()
    # Sort by family then type order
    type_order_map = {'base': 0, 'instruct': 1, 'reasoning': 2}
    models['type_order'] = models['model_type'].map(type_order_map)
    models = models.sort_values(['family', 'type_order']).reset_index(drop=True)

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 3 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    type_colors = {'base': '#4C72B0', 'instruct': '#DD8452', 'reasoning': '#55A868'}

    for i, (_, model_row) in enumerate(models.iterrows()):
        mname = model_row['model_name']
        mtype = model_row['model_type']
        color = type_colors.get(mtype, 'gray')
        df_model = df_all[df_all['model_name'] == mname]

        for j, cat_name in enumerate(['public_issues', 'private_life']):
            ax = axes[i, j]
            df_cat = df_model[df_model['category'] == cat_name]
            df_pol = pol_data[cat_name]

            df_merged = df_cat.merge(
                df_pol[['variable', 'polarization']].rename(
                    columns={'variable': 'Topic', 'polarization': 'GSS_Polarization'}
                ),
                on='Topic', how='inner'
            )

            if len(df_merged) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            ax.scatter(df_merged['GSS_Polarization'], df_merged[col],
                       alpha=0.5, s=20, color=color)

            r, _ = pearsonr(df_merged[col], df_merged['GSS_Polarization'])
            # Regression line
            z = np.polyfit(df_merged['GSS_Polarization'], df_merged[col], 1)
            xline = np.linspace(df_merged['GSS_Polarization'].min(),
                                df_merged['GSS_Polarization'].max(), 100)
            ax.plot(xline, np.polyval(z, xline), color=color, linewidth=1.5, alpha=0.7)

            ax.set_title(f'{mname} | {cat_name.replace("_"," ")} (r={r:.3f})', fontsize=9)
            ax.set_xlabel('GSS Polarization', fontsize=8)
            ax.set_ylabel(f'Avg Mahal PCA-{pca_dim}', fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  >> Saved scatter grid: {output_path}', flush=True)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Base vs Instruct vs Reasoning Model Comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--families', type=str, nargs='*', default=None,
                        help='Subset of families to run (e.g., Qwen3-4B Llama-3.1-8B)')
    parser.add_argument('--model-types', type=str, nargs='*', default=None,
                        help='Subset of model types to run (e.g., base instruct reasoning)')
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
    parser.add_argument('--batch-size', type=int, default=80)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--pca-dims', type=int, nargs='+', default=[1, 3, 5, 10, 15])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--min-dem', type=int, default=100)
    parser.add_argument('--min-rep', type=int, default=100)
    parser.add_argument('--min-total', type=int, default=200)
    parser.add_argument('--exclude-public', type=str, nargs='*',
                        default=['hubbywk1', 'racdif1', 'racdif2', 'racdif3', 'racdif4',
                                 'workwhts', 'wlthwhts', 'intlwhts'])
    parser.add_argument('--exclude-private', type=str, nargs='*',
                        default=['reborn', 'marwht', 'helpful', 'helpfulnv', 'helpfulv'])
    parser.add_argument('--plot-pca-dim', type=int, default=15)
    parser.add_argument('--plot-centroid', type=str, default='mean',
                        choices=['mean', 'median'])
    parser.add_argument('--skip-plots', action='store_true')
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ---- Configuration summary ----
    families_to_run = args.families or list(MODEL_FAMILIES.keys())
    print('=' * 70)
    print('BASE vs INSTRUCT vs REASONING MODEL COMPARISON')
    print('=' * 70)
    print(f'  Timestamp: {timestamp}')
    print(f'  Families:  {families_to_run}')
    print(f'  PCA dims:  {args.pca_dims}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Output dir: {output_dir}')

    total_models = sum(
        len(MODEL_FAMILIES[f]) for f in families_to_run if f in MODEL_FAMILIES
    )
    print(f'  Total models to run: {total_models}')
    print()

    for fam in families_to_run:
        if fam not in MODEL_FAMILIES:
            print(f'  WARNING: Family "{fam}" not found in MODEL_FAMILIES, skipping')
            continue
        print(f'  [{fam}]')
        for variant, cfg in MODEL_FAMILIES[fam].items():
            print(f'    {variant:12s} -> {cfg["path"]}')
    print('=' * 70, flush=True)

    # ---- Load shared data ----
    print('\nLoading shared data...')

    # Politicians
    df_politicians = load_politicians(args.politician_csv)
    df_politicians = df_politicians[df_politicians['party_code'].isin([100, 200])].dropna(subset=['fullname'])
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = df_politicians['party_code'].values
    n_dem = int((politician_labels == 100).sum())
    n_rep = int((politician_labels == 200).sum())
    print(f'  Politicians: {len(politician_names)} (D={n_dem}, R={n_rep})')

    # Topics
    pub_topics, _ = load_topics_from_csv(args.public_topics)
    priv_topics, _ = load_topics_from_csv(args.private_topics)

    # Polarization data (for filtering topics)
    pol_pub = load_polarization_data(args.public_polarization)
    pol_priv = load_polarization_data(args.private_polarization)

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

    pub_valid = set(pol_pub['variable'])
    priv_valid = set(pol_priv['variable'])
    pub_topics = {k: v for k, v in pub_topics.items() if k in pub_valid}
    priv_topics = {k: v for k, v in priv_topics.items() if k in priv_valid}

    print(f'  Public issues topics:  {len(pub_topics)}')
    print(f'  Private life topics:   {len(priv_topics)}')

    all_topics = {
        'public_issues': pub_topics,
        'private_life': priv_topics,
    }
    pol_data = {
        'public_issues': pol_pub,
        'private_life': pol_priv,
    }

    # ---- Run models ----
    all_model_results = []

    for fam_name in families_to_run:
        if fam_name not in MODEL_FAMILIES:
            continue
        family = MODEL_FAMILIES[fam_name]

        for variant_name, variant_cfg in family.items():
            model_type = variant_cfg['type']

            # Skip if model type not in filter
            if args.model_types and model_type not in args.model_types:
                continue

            model_name = f'{fam_name}_{variant_name}'
            model_path = variant_cfg['path']
            # Per-model batch size (falls back to CLI --batch-size)
            effective_batch_size = variant_cfg.get('batch_size', args.batch_size)
            system_override = variant_cfg.get('system_override', None)

            df_model = run_single_model(
                model_path=model_path,
                model_name=model_name,
                model_type=model_type,
                family_name=fam_name,
                all_topics=all_topics,
                politician_names=politician_names,
                politician_labels=politician_labels,
                pca_dims=args.pca_dims,
                batch_size=effective_batch_size,
                max_length=args.max_length,
                output_dir=output_dir,
                timestamp=timestamp,
                system_override=system_override,
            )
            all_model_results.append(df_model)

            # Save running aggregate checkpoint
            df_agg = pd.concat(all_model_results, ignore_index=True)
            agg_path = output_dir / f'comparison_all_{timestamp}_checkpoint.pkl'
            df_agg.to_pickle(agg_path)
            print(f'  >> Checkpoint ({len(df_agg)} total rows): {agg_path}', flush=True)

    # ---- Aggregate results ----
    print(f'\n{"="*70}')
    print('AGGREGATING RESULTS')
    print(f'{"="*70}')

    df_all = pd.concat(all_model_results, ignore_index=True)
    print(f'Total results: {len(df_all)} rows')

    # Save detail CSV/PKL
    detail_csv = output_dir / f'comparison_detail_{timestamp}.csv'
    detail_pkl = output_dir / f'comparison_detail_{timestamp}.pkl'
    df_all.to_csv(detail_csv, index=False)
    df_all.to_pickle(detail_pkl)
    print(f'  >> Saved: {detail_csv}')
    print(f'  >> Saved: {detail_pkl}')

    # ---- Correlations (all topics) ----
    print(f'\n{"="*70}')
    print('CORRELATION ANALYSIS (all topics)')
    print(f'{"="*70}')

    df_corr_all = compute_comparison_correlations(df_all, pol_data, args.pca_dims)
    corr_all_path = output_dir / f'comparison_correlations_{timestamp}.csv'
    df_corr_all.to_csv(corr_all_path, index=False)
    print(f'  >> Saved: {corr_all_path}')
    print(df_corr_all.to_string(index=False))

    # ---- Correlations (filtered) ----
    print(f'\n{"="*70}')
    print('CORRELATION ANALYSIS (filtered)')
    print(f'  Excluded public:  {args.exclude_public}')
    print(f'  Excluded private: {args.exclude_private}')
    print(f'{"="*70}')

    df_corr_filt = compute_comparison_correlations(
        df_all, pol_data, args.pca_dims,
        exclude_pub=args.exclude_public,
        exclude_priv=args.exclude_private,
    )
    corr_filt_path = output_dir / f'comparison_filtered_{timestamp}.csv'
    df_corr_filt.to_csv(corr_filt_path, index=False)
    print(f'  >> Saved: {corr_filt_path}')
    print(df_corr_filt.to_string(index=False))

    # ---- Plots ----
    if not args.skip_plots:
        print(f'\n{"="*70}')
        print('GENERATING PLOTS')
        print(f'{"="*70}')

        bar_path = output_dir / f'comparison_bar_{timestamp}.png'
        plot_comparison_bar_chart(
            df_corr_filt, args.plot_pca_dim, args.plot_centroid, bar_path
        )

        scatter_path = output_dir / f'comparison_scatter_{timestamp}.png'
        plot_scatter_grid(
            df_all, pol_data, args.plot_pca_dim, args.plot_centroid, scatter_path
        )

    # ---- Done ----
    print(f'\n{"="*70}')
    print('DONE')
    print(f'{"="*70}')
    print(f'  Detail CSV:       {detail_csv}')
    print(f'  Correlations:     {corr_all_path}')
    print(f'  Filtered:         {corr_filt_path}')
    if not args.skip_plots:
        print(f'  Bar chart:        {bar_path}')
        print(f'  Scatter grid:     {scatter_path}')


if __name__ == '__main__':
    main()
