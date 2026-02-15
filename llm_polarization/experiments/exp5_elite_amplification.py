"""
Experiment 5: Elite Amplification

Research Question:
    Do LLMs amplify real-world partisan differences? Is the model-internal
    separation proportional to actual GSS-measured polarization, or do models
    exaggerate/compress it?

Hypotheses:
    H5a: Instruct models amplify (ratio > 1) more than base models
    H5b: Amplification is stronger for high-profile culture war topics
    H5c: Reasoning models show less amplification than instruct (closer to reality)

Method:
    - Use the 30 most polarized GSS topics (exp5_polarized_topics.json)
    - For each model × topic: compute Mahalanobis distance (model signal)
    - For each topic: compute GSS |mean_dem - mean_rep| (real signal)
    - Amplification ratio = model_distance / gss_distance
    - Also compute rank-order correlation (Spearman rho)

Analysis:
    - Scatter: model distance vs GSS polarization (with regression line)
    - Amplification ratio distribution by model type
    - Identify topics with highest amplification (model creates false separation)
    - Identify topics with lowest amplification (model misses real differences)

Runtime: ~3 hours on H100
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict

import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind
import torch

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    set_random_seeds,
    load_polarization_data,
    compute_pca_and_distance,
    save_checkpoint,
    setup_plot_style,
    save_figure,
    MODEL_FAMILIES,
    RESULTS_DIR,
    TOPIC_LISTS_DIR,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from config import SYSTEM_MSG_POLITICIAN

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp5"
PCA_DIMS = [5, 10, 15]
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/data/HS116_members_fullname.csv"

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 5: Elite Amplification."""

    print("="*80)
    print("EXPERIMENT 5: ELITE AMPLIFICATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics (30 most polarized)
    print("Loading polarized topics...")
    with open(TOPIC_LISTS_DIR / "exp5_polarized_topics.json") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} polarized topics")

    # Load GSS data
    print("Loading GSS polarization data...")
    gss_df = load_polarization_data()

    # Build GSS lookup: topic -> polarization metrics
    gss_lookup = {}
    for _, row in gss_df.iterrows():
        gss_lookup[row['variable']] = {
            'polarization': row['polarization'],
            'mean_dem': row['mean_dem'],
            'mean_rep': row['mean_rep'],
            'mean_diff': abs(row['mean_dem'] - row['mean_rep']),
            'n_total': row['n_total'],
        }

    # Load politicians
    print("\nLoading politicians...")
    df_politicians = load_politicians(POLITICIAN_CSV)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)
    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Prompt templates
    BASE_TEMPLATE = "{name} makes a statement on {topic}:"
    INSTRUCT_TEMPLATE_KEY = "default"

    # Results storage
    all_results = []

    # Run all models - load once, process inline to save memory
    for family_name, family_config in MODEL_FAMILIES.items():
        print(f"\n{'='*80}")
        print(f"FAMILY: {family_name}")
        print(f"{'='*80}")

        for variant_name, variant_cfg in family_config.items():
            model_path = variant_cfg['path']
            model_type = variant_cfg['type']
            batch_size = variant_cfg['batch_size']
            model_name = f"{family_name}_{variant_name}"

            print(f"\n--- Running {model_name} ({model_type}) ---")

            # Load model ONCE
            model, tokenizer = load_model(model_path)
            if model_type == "base":
                tokenizer.chat_template = None

            for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                print(f"  [{topic_idx}/{len(topics)}] {topic_name}")

                # Generate prompts
                if model_type == "base":
                    prompts = [BASE_TEMPLATE.format(name=name, topic=topic_desc)
                               for name in politician_names]
                    system_msg = ""
                else:
                    template = POLITICIAN_TEMPLATES[INSTRUCT_TEMPLATE_KEY]
                    prompts = generate_politician_prompts(
                        topic_desc, politician_names, template=template
                    )
                    system_msg = SYSTEM_MSG_POLITICIAN

                # Extract activations
                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=batch_size, max_length=MAX_LENGTH,
                )

                # Compute PCA immediately and discard activations
                for pca_dim in PCA_DIMS:
                    pca_result = compute_pca_and_distance(
                        activations, politician_labels, pca_dim=pca_dim
                    )

                    result = {
                        'family': family_name,
                        'variant': variant_name,
                        'model_name': model_name,
                        'model_type': model_type,
                        'topic_name': topic_name,
                        'pca_dim': pca_dim,
                        'mahalanobis_dist': pca_result['mahalanobis_dist'],
                        'variance_explained': pca_result['variance_explained'],
                    }

                    # Add GSS data
                    if topic_name in gss_lookup:
                        gss = gss_lookup[topic_name]
                        result['gss_polarization'] = gss['polarization']
                        result['gss_mean_diff'] = gss['mean_diff']

                        if gss['mean_diff'] > 0:
                            result['amplification_ratio'] = pca_result['mahalanobis_dist'] / gss['mean_diff']
                        else:
                            result['amplification_ratio'] = np.nan
                    else:
                        result['gss_polarization'] = np.nan
                        result['gss_mean_diff'] = np.nan
                        result['amplification_ratio'] = np.nan

                    all_results.append(result)

                del activations
                torch.cuda.empty_cache()
                gc.collect()

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Per-model checkpoint
            model_results = [r for r in all_results if r['model_name'] == model_name]
            save_checkpoint(model_results, EXPERIMENT_NAME, model_name)

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"exp5_amplification_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # Focus on PCA=15 for main analysis
    df15 = df[df['pca_dim'] == 15].copy()
    df15_valid = df15.dropna(subset=['gss_polarization', 'amplification_ratio'])

    # Summary by model type
    print("\n--- Amplification Ratio by Model Type ---")
    amp_summary = df15_valid.groupby('model_type')['amplification_ratio'].agg(['mean', 'median', 'std'])
    print(amp_summary.round(3))

    # Correlation with GSS
    print("\n--- Correlation: Model Distance vs GSS Polarization ---")
    for model_type in ['base', 'instruct', 'reasoning']:
        df_mt = df15_valid[df15_valid['model_type'] == model_type]
        if len(df_mt) > 3:
            r_pearson, p_pearson = pearsonr(df_mt['mahalanobis_dist'], df_mt['gss_polarization'])
            r_spearman, p_spearman = spearmanr(df_mt['mahalanobis_dist'], df_mt['gss_polarization'])
            print(f"  {model_type}: Pearson r={r_pearson:.3f} (p={p_pearson:.4f}), "
                  f"Spearman rho={r_spearman:.3f} (p={p_spearman:.4f})")

    # Top amplified topics
    print("\n--- Top 5 Most Amplified Topics (instruct models, PCA=15) ---")
    df_instruct = df15_valid[df15_valid['model_type'] == 'instruct']
    top_amp = df_instruct.groupby('topic_name')['amplification_ratio'].mean().nlargest(5)
    for topic, ratio in top_amp.items():
        print(f"  {topic}: amplification={ratio:.2f}")

    # Bottom amplified topics
    print("\n--- Top 5 Least Amplified Topics (instruct models, PCA=15) ---")
    bot_amp = df_instruct.groupby('topic_name')['amplification_ratio'].mean().nsmallest(5)
    for topic, ratio in bot_amp.items():
        print(f"  {topic}: amplification={ratio:.2f}")

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Scatter - Model distance vs GSS polarization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, model_type in enumerate(['base', 'instruct', 'reasoning']):
        ax = axes[idx]
        df_mt = df15_valid[df15_valid['model_type'] == model_type]

        if len(df_mt) == 0:
            ax.set_title(f'{model_type} (no data)')
            continue

        for family in df_mt['family'].unique():
            df_fam = df_mt[df_mt['family'] == family]
            ax.scatter(df_fam['gss_polarization'], df_fam['mahalanobis_dist'],
                      label=family, alpha=0.6, s=40)

        # Regression line
        if len(df_mt) > 3:
            z = np.polyfit(df_mt['gss_polarization'], df_mt['mahalanobis_dist'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df_mt['gss_polarization'].min(), df_mt['gss_polarization'].max(), 100)
            ax.plot(x_range, p(x_range), 'k--', alpha=0.5)

            r, p_val = pearsonr(df_mt['gss_polarization'], df_mt['mahalanobis_dist'])
            ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3f}',
                   transform=ax.transAxes, va='top', fontsize=9)

        ax.set_xlabel('GSS Polarization')
        ax.set_ylabel('Model Mahalanobis Distance')
        ax.set_title(f'{model_type.title()} Models')
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'scatter_gss_vs_model')
    plt.close()

    # Plot 2: Amplification ratio by model type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.boxplot(data=df15_valid, x='model_type', y='amplification_ratio', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='No amplification')
    ax.set_title('Amplification Ratio by Model Type')
    ax.set_ylabel('Amplification Ratio (model / GSS)')
    ax.set_xlabel('Model Type')
    ax.legend()

    ax = axes[1]
    sns.barplot(data=df15_valid, x='family', y='amplification_ratio', hue='model_type', ax=ax,
                hue_order=['base', 'instruct', 'reasoning'])
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Amplification by Family')
    ax.set_ylabel('Amplification Ratio')
    ax.set_xlabel('Model Family')
    ax.legend(title='Model Type')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'amplification_ratios')
    plt.close()

    # Plot 3: Per-topic amplification heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    pivot = df15_valid.pivot_table(
        values='amplification_ratio',
        index='topic_name',
        columns='model_name',
        aggfunc='mean'
    )
    pivot = pivot.sort_values(by=pivot.columns[0], ascending=False) if len(pivot.columns) > 0 else pivot

    sns.heatmap(pivot, ax=ax, cmap='RdYlBu_r', center=1.0,
                annot=False, xticklabels=True, yticklabels=True)
    ax.set_title('Amplification Ratio by Topic × Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Topic')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'amplification_heatmap')
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 5 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_experiment()
