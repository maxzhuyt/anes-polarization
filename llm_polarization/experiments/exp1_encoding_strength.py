"""
Experiment 1: Prompt-End Encoding Strength

Research Question:
    Do reasoning models encode partisan information more weakly than base/instruct
    models at prompt completion (before generation)?

Hypotheses:
    H1a: PC1 variance explained: base > instruct > reasoning (Cohen's d > 0.5)
    H1b: Mahalanobis distance: base > instruct > reasoning (Cohen's d > 0.5)

Method:
    - Sample: 30 public + 30 private topics (pre-specified, seed=42)
    - Models: 9 (Qwen, Llama, Gemma × base/instruct/reasoning)
    - Extract activations at final prompt token
    - Compute PCA (dims 5, 10, 15) + Mahalanobis distance
    - Compare across model types

Analysis:
    - Repeated-measures ANOVA: DV ~ model_type × category
    - Tukey HSD post-hoc tests
    - Report Cohen's d effect sizes
    - FDR correction (Benjamini-Hochberg q < 0.05)

Runtime: ~2.5 hours on H100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    set_random_seeds,
    load_all_topics,
    load_polarization_data,
    compute_pca_and_distance,
    save_checkpoint,
    setup_plot_style,
    save_figure,
    MODEL_FAMILIES,
    RESULTS_DIR,
    TOPIC_LISTS_DIR,
)

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from config import SYSTEM_MSG_POLITICIAN

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp1"
PCA_DIMS = [5, 10, 15]
MAX_LENGTH = 128
REPLICATION = False  # Set to True for Exp1R

# =============================================================================
# Topic Loading
# =============================================================================

def load_experiment_topics(replication: bool = False) -> Dict[str, Dict]:
    """Load pre-specified topic lists for this experiment."""
    prefix = "exp1r" if replication else "exp1"

    with open(TOPIC_LISTS_DIR / f"{prefix}_public.json") as f:
        public_topics = json.load(f)

    with open(TOPIC_LISTS_DIR / f"{prefix}_private.json") as f:
        private_topics = json.load(f)

    return {
        'public_issues': public_topics,
        'private_life': private_topics
    }

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(replication: bool = False):
    """Run Experiment 1 (or 1R if replication=True)."""
    exp_name = "exp1r" if replication else "exp1"

    print("="*80)
    print(f"EXPERIMENT 1{'R' if replication else ''}: PROMPT-END ENCODING STRENGTH")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Set random seeds
    set_random_seeds(42)

    # Load topics
    print("Loading topics...")
    topics_by_category = load_experiment_topics(replication)
    total_topics = sum(len(topics) for topics in topics_by_category.values())
    print(f"Loaded {total_topics} topics:")
    for category, topics in topics_by_category.items():
        print(f"  {category}: {len(topics)} topics")

    # Load politicians (same as run_model_comparison.py)
    print("\nLoading politicians...")
    politician_csv = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
    df_politicians = load_politicians(politician_csv)
    politician_names = df_politicians['fullname'].tolist()
    # party_code: 100=Democrat, 200=Republican -> labels: 0=Democrat, 1=Republican
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)

    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Results storage
    all_results = []

    # Prompt templates (same as shared_utils / run_model_comparison)
    BASE_TEMPLATES = {
        "public_issues": "{name} makes a statement on {topic}:",
        "private_life":  "When asked about {topic}, {name} says",
    }
    INSTRUCT_TEMPLATE_KEYS = {
        "public_issues": "default",
        "private_life":  "opinion",
    }

    # Run all models - load each model ONCE and process all categories/topics inline
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

            # Load model ONCE for all categories
            model, tokenizer = load_model(model_path)
            if model_type == "base":
                tokenizer.chat_template = None
                print("Disabled chat template for base model")

            for category, topics in topics_by_category.items():
                print(f"\n  Category: {category}")

                for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                    print(f"  [{topic_idx}/{len(topics)}] {topic_name}")

                    # Generate prompts
                    if model_type == "base":
                        template = BASE_TEMPLATES[category]
                        prompts = [template.format(name=name, topic=topic_desc)
                                   for name in politician_names]
                        system_msg = ""
                    else:
                        template_key = INSTRUCT_TEMPLATE_KEYS[category]
                        template = POLITICIAN_TEMPLATES[template_key]
                        prompts = generate_politician_prompts(
                            topic_desc, politician_names, template=template
                        )
                        system_msg = SYSTEM_MSG_POLITICIAN

                    # Extract activations
                    activations = extract_heads_batched(
                        model, tokenizer, prompts, system_msg,
                        batch_size=batch_size, max_length=MAX_LENGTH,
                    )

                    # Compute PCA and Mahalanobis IMMEDIATELY, then discard activations
                    for pca_dim in PCA_DIMS:
                        pca_results = compute_pca_and_distance(
                            activations,
                            politician_labels,
                            pca_dim=pca_dim
                        )

                        all_results.append({
                            'family': family_name,
                            'variant': variant_name,
                            'model_name': model_name,
                            'model_type': model_type,
                            'category': category,
                            'topic_name': topic_name,
                            'pca_dim': pca_dim,
                            'variance_explained_pc1': pca_results['variance_explained'],
                            'mahalanobis_dist': pca_results['mahalanobis_dist'],
                            'n_politicians': len(politician_names),
                        })

                    # Free activation memory immediately
                    del activations
                    torch.cuda.empty_cache()
                    gc.collect()

                print(f"  Processed {len(topics)} topics for {category}")

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Save per-model checkpoint
            model_results = [r for r in all_results if r['model_name'] == model_name]
            checkpoint_path = save_checkpoint(
                model_results,
                exp_name,
                model_name
            )
            print(f"\nSaved checkpoint: {checkpoint_path}")

    # ==========================================================================
    # Analysis and Visualization
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS AND VISUALIZATION")
    print(f"{'='*80}\n")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"{exp_name}_detail_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results: {csv_path}")

    # Aggregate by model and category
    df_agg = df.groupby(['family', 'variant', 'model_name', 'model_type', 'category', 'pca_dim']).agg({
        'variance_explained_pc1': ['mean', 'std'],
        'mahalanobis_dist': ['mean', 'std']
    }).reset_index()

    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

    # Save aggregated results
    agg_csv_path = RESULTS_DIR / f"{exp_name}_aggregated_{timestamp}.csv"
    df_agg.to_csv(agg_csv_path, index=False)
    print(f"Saved aggregated results: {agg_csv_path}")

    # Statistical tests
    print("\n--- Statistical Tests ---")

    for pca_dim in PCA_DIMS:
        print(f"\nPCA Dimension: {pca_dim}")

        df_dim = df[df['pca_dim'] == pca_dim].copy()

        # ANOVA: variance_explained_pc1 ~ model_type
        print("\n  Variance Explained (PC1):")
        base_var = df_dim[df_dim['model_type'] == 'base']['variance_explained_pc1'].values
        instruct_var = df_dim[df_dim['model_type'] == 'instruct']['variance_explained_pc1'].values
        reasoning_var = df_dim[df_dim['model_type'] == 'reasoning']['variance_explained_pc1'].dropna().values

        if len(reasoning_var) > 0:
            f_stat, p_val = f_oneway(base_var, instruct_var, reasoning_var)
            print(f"    ANOVA: F={f_stat:.4f}, p={p_val:.4e}")

        # Pairwise t-tests with FDR correction
        print("    Pairwise comparisons:")
        comparisons = [
            ('base', 'instruct'),
            ('base', 'reasoning'),
            ('instruct', 'reasoning')
        ]

        p_values = []
        for type1, type2 in comparisons:
            data1 = df_dim[df_dim['model_type'] == type1]['variance_explained_pc1'].values
            data2 = df_dim[df_dim['model_type'] == type2]['variance_explained_pc1'].dropna().values

            if len(data1) > 0 and len(data2) > 0:
                t_stat, p_val = ttest_ind(data1, data2)
                p_values.append(p_val)

                # Cohen's d
                pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

                print(f"      {type1} vs {type2}: t={t_stat:.4f}, p={p_val:.4e}, d={cohens_d:.3f}")
            else:
                p_values.append(1.0)

        # FDR correction
        if len(p_values) > 0:
            reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            print("    FDR-corrected p-values:")
            for (type1, type2), p_corr, rej in zip(comparisons, p_corrected, reject):
                print(f"      {type1} vs {type2}: p_adj={p_corr:.4e}, reject={rej}")

        # Same for Mahalanobis distance
        print("\n  Mahalanobis Distance:")
        base_dist = df_dim[df_dim['model_type'] == 'base']['mahalanobis_dist'].values
        instruct_dist = df_dim[df_dim['model_type'] == 'instruct']['mahalanobis_dist'].values
        reasoning_dist = df_dim[df_dim['model_type'] == 'reasoning']['mahalanobis_dist'].dropna().values

        if len(reasoning_dist) > 0:
            f_stat, p_val = f_oneway(base_dist, instruct_dist, reasoning_dist)
            print(f"    ANOVA: F={f_stat:.4f}, p={p_val:.4e}")

        p_values = []
        print("    Pairwise comparisons:")
        for type1, type2 in comparisons:
            data1 = df_dim[df_dim['model_type'] == type1]['mahalanobis_dist'].values
            data2 = df_dim[df_dim['model_type'] == type2]['mahalanobis_dist'].dropna().values

            if len(data1) > 0 and len(data2) > 0:
                t_stat, p_val = ttest_ind(data1, data2)
                p_values.append(p_val)

                pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

                print(f"      {type1} vs {type2}: t={t_stat:.4f}, p={p_val:.4e}, d={cohens_d:.3f}")
            else:
                p_values.append(1.0)

        if len(p_values) > 0:
            reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            print("    FDR-corrected p-values:")
            for (type1, type2), p_corr, rej in zip(comparisons, p_corrected, reject):
                print(f"      {type1} vs {type2}: p_adj={p_corr:.4e}, reject={rej}")

    # Plots
    print("\n--- Generating Plots ---")
    setup_plot_style()

    # Plot 1: Variance explained by model type
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, pca_dim in enumerate(PCA_DIMS):
        ax = axes[idx]
        df_dim = df[df['pca_dim'] == pca_dim].copy()

        # Box plot
        df_dim_plot = df_dim.groupby(['model_type', 'model_name']).agg({
            'variance_explained_pc1': 'mean'
        }).reset_index()

        sns.boxplot(data=df_dim_plot, x='model_type', y='variance_explained_pc1', ax=ax,
                    order=['base', 'instruct', 'reasoning'])
        sns.swarmplot(data=df_dim_plot, x='model_type', y='variance_explained_pc1', ax=ax,
                      color='black', alpha=0.5, order=['base', 'instruct', 'reasoning'])

        ax.set_title(f'PCA Dim = {pca_dim}')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('PC1 Variance Explained')
        ax.set_ylim(0, None)

    plt.tight_layout()
    save_figure(fig, exp_name, 'variance_explained')
    plt.close()

    # Plot 2: Mahalanobis distance by model type
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, pca_dim in enumerate(PCA_DIMS):
        ax = axes[idx]
        df_dim = df[df['pca_dim'] == pca_dim].copy()

        df_dim_plot = df_dim.groupby(['model_type', 'model_name']).agg({
            'mahalanobis_dist': 'mean'
        }).reset_index()

        sns.boxplot(data=df_dim_plot, x='model_type', y='mahalanobis_dist', ax=ax,
                    order=['base', 'instruct', 'reasoning'])
        sns.swarmplot(data=df_dim_plot, x='model_type', y='mahalanobis_dist', ax=ax,
                      color='black', alpha=0.5, order=['base', 'instruct', 'reasoning'])

        ax.set_title(f'PCA Dim = {pca_dim}')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Mahalanobis Distance')
        ax.set_ylim(0, None)

    plt.tight_layout()
    save_figure(fig, exp_name, 'mahalanobis_dist')
    plt.close()

    # Plot 3: By category and model type
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for cat_idx, category in enumerate(['public_issues', 'private_life']):
        for dim_idx, pca_dim in enumerate(PCA_DIMS):
            ax = axes[cat_idx, dim_idx]

            df_subset = df[(df['category'] == category) & (df['pca_dim'] == pca_dim)].copy()

            df_plot = df_subset.groupby(['model_type', 'model_name']).agg({
                'mahalanobis_dist': 'mean'
            }).reset_index()

            sns.barplot(data=df_plot, x='model_type', y='mahalanobis_dist', ax=ax,
                       order=['base', 'instruct', 'reasoning'], errorbar='sd')

            ax.set_title(f'{category.replace("_", " ").title()}, PCA={pca_dim}')
            ax.set_xlabel('Model Type')
            ax.set_ylabel('Mahalanobis Distance')

    plt.tight_layout()
    save_figure(fig, exp_name, 'distance_by_category')
    plt.close()

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1{'R' if replication else ''} COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

# =============================================================================
# Command-Line Interface
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Encoding Strength")
    parser.add_argument(
        "--replication",
        action="store_true",
        help="Run Experiment 1R (replication with different topic sample)"
    )

    args = parser.parse_args()

    run_experiment(replication=args.replication)
