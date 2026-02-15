"""
Experiment 6: False Polarization Detection

Research Question:
    Do LLMs create partisan separation on topics where real-world data shows
    minimal or no polarization? Which topics are most susceptible to false
    polarization?

Hypotheses:
    H6a: Models show significant Mahalanobis distance even on low-GSS-polarization topics
    H6b: False polarization rate is higher for instruct > base > reasoning
    H6c: Private life topics are more susceptible to false polarization than public issues

Method:
    - Use 40 topics spanning the full polarization spectrum (exp6_overlap_topics.json)
    - For each model × topic: compute Mahalanobis distance
    - Define "false polarization" as: model_distance > threshold AND gss_polarization < median
    - Compute false positive rate, false negative rate

Analysis:
    - ROC curve: can model distance predict actual GSS polarization?
    - Identify systematically over/under-polarized topics
    - Compare false polarization rates across model types

Runtime: ~3 hours on H100
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from sklearn.metrics import roc_curve, auc, precision_recall_curve
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

EXPERIMENT_NAME = "exp6"
PCA_DIM = 15
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 6: False Polarization Detection."""

    print("="*80)
    print("EXPERIMENT 6: FALSE POLARIZATION DETECTION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics (spanning full polarization range)
    print("Loading overlap topics...")
    with open(TOPIC_LISTS_DIR / "exp6_overlap_topics.json") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} topics")

    # Load GSS data
    print("Loading GSS polarization data...")
    gss_df = load_polarization_data()

    gss_lookup = {}
    for _, row in gss_df.iterrows():
        gss_lookup[row['variable']] = {
            'polarization': row['polarization'],
            'mean_dem': row['mean_dem'],
            'mean_rep': row['mean_rep'],
            'mean_diff': abs(row['mean_dem'] - row['mean_rep']),
        }

    # Determine polarization median for thresholding
    topic_polarizations = [gss_lookup[t]['polarization'] for t in topics if t in gss_lookup]
    median_pol = np.median(topic_polarizations) if topic_polarizations else 0.5
    print(f"Median GSS polarization: {median_pol:.4f}")

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

    # Run all models - load once, process inline
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

                # Compute PCA immediately
                pca_result = compute_pca_and_distance(
                    activations, politician_labels, pca_dim=PCA_DIM
                )

                result = {
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': topic_name,
                    'mahalanobis_dist': pca_result['mahalanobis_dist'],
                    'variance_explained': pca_result['variance_explained'],
                }

                if topic_name in gss_lookup:
                    gss = gss_lookup[topic_name]
                    result['gss_polarization'] = gss['polarization']
                    result['gss_mean_diff'] = gss['mean_diff']
                    result['gss_high'] = int(gss['polarization'] >= median_pol)
                else:
                    result['gss_polarization'] = np.nan
                    result['gss_mean_diff'] = np.nan
                    result['gss_high'] = np.nan

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
    df_valid = df.dropna(subset=['gss_polarization'])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"exp6_false_polarization_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # Median-split analysis
    median_model_dist = df_valid.groupby('model_type')['mahalanobis_dist'].median()
    print("\n--- Median Model Distance by Type ---")
    print(median_model_dist.round(3))

    # False polarization: high model distance on low-GSS topics
    print("\n--- False Polarization Analysis ---")
    for model_type in ['base', 'instruct', 'reasoning']:
        df_mt = df_valid[df_valid['model_type'] == model_type]
        if len(df_mt) == 0:
            continue

        # Use per-model-type median distance as threshold
        dist_median = df_mt['mahalanobis_dist'].median()

        # Low GSS topics where model shows high distance
        low_gss = df_mt[df_mt['gss_high'] == 0]
        false_pos = low_gss[low_gss['mahalanobis_dist'] > dist_median]
        false_pos_rate = len(false_pos) / len(low_gss) if len(low_gss) > 0 else 0

        # High GSS topics where model shows low distance
        high_gss = df_mt[df_mt['gss_high'] == 1]
        false_neg = high_gss[high_gss['mahalanobis_dist'] <= dist_median]
        false_neg_rate = len(false_neg) / len(high_gss) if len(high_gss) > 0 else 0

        print(f"  {model_type}: FP rate={false_pos_rate:.3f}, FN rate={false_neg_rate:.3f}")

    # ROC analysis: can model distance predict GSS polarization category?
    print("\n--- ROC Analysis ---")
    for model_type in ['base', 'instruct', 'reasoning']:
        df_mt = df_valid[df_valid['model_type'] == model_type].dropna(subset=['gss_high'])
        if len(df_mt) < 10:
            continue

        fpr, tpr, _ = roc_curve(df_mt['gss_high'], df_mt['mahalanobis_dist'])
        roc_auc_val = auc(fpr, tpr)
        print(f"  {model_type}: AUC = {roc_auc_val:.3f}")

    # Rank-order correlation
    print("\n--- Rank-Order Correlation (model dist vs GSS pol) ---")
    for model_type in ['base', 'instruct', 'reasoning']:
        df_mt = df_valid[df_valid['model_type'] == model_type]
        if len(df_mt) > 3:
            # Average across families for same topic
            avg_by_topic = df_mt.groupby('topic_name').agg({
                'mahalanobis_dist': 'mean',
                'gss_polarization': 'first'
            }).reset_index()
            rho, p = spearmanr(avg_by_topic['mahalanobis_dist'], avg_by_topic['gss_polarization'])
            print(f"  {model_type}: Spearman rho={rho:.3f}, p={p:.4f}")

    # Most falsely polarized topics
    print("\n--- Most Falsely Polarized Topics (low GSS, high model distance) ---")
    low_gss_topics = df_valid[df_valid['gss_high'] == 0]
    if len(low_gss_topics) > 0:
        top_false = low_gss_topics.groupby('topic_name').agg({
            'mahalanobis_dist': 'mean',
            'gss_polarization': 'first',
        }).sort_values('mahalanobis_dist', ascending=False).head(5)
        for topic, row in top_false.iterrows():
            print(f"  {topic}: model_dist={row['mahalanobis_dist']:.3f}, gss_pol={row['gss_polarization']:.4f}")

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Model distance vs GSS polarization with quadrants
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, model_type in enumerate(['base', 'instruct', 'reasoning']):
        ax = axes[idx]
        df_mt = df_valid[df_valid['model_type'] == model_type]

        if len(df_mt) == 0:
            ax.set_title(f'{model_type} (no data)')
            continue

        # Average across families for cleaner plot
        avg = df_mt.groupby('topic_name').agg({
            'mahalanobis_dist': 'mean',
            'gss_polarization': 'first',
        }).reset_index()

        ax.scatter(avg['gss_polarization'], avg['mahalanobis_dist'], alpha=0.6, s=50)

        # Quadrant lines
        ax.axhline(avg['mahalanobis_dist'].median(), color='gray', linestyle='--', alpha=0.3)
        ax.axvline(median_pol, color='gray', linestyle='--', alpha=0.3)

        # Label quadrants
        ax.text(0.02, 0.98, 'False\nPolarization', transform=ax.transAxes,
               va='top', fontsize=8, color='red', alpha=0.5)
        ax.text(0.98, 0.98, 'True\nPolarization', transform=ax.transAxes,
               va='top', ha='right', fontsize=8, color='green', alpha=0.5)
        ax.text(0.02, 0.02, 'True\nConsensus', transform=ax.transAxes,
               va='bottom', fontsize=8, color='green', alpha=0.5)
        ax.text(0.98, 0.02, 'Missed\nPolarization', transform=ax.transAxes,
               va='bottom', ha='right', fontsize=8, color='orange', alpha=0.5)

        # Regression line
        if len(avg) > 3:
            r, p = pearsonr(avg['gss_polarization'], avg['mahalanobis_dist'])
            z = np.polyfit(avg['gss_polarization'], avg['mahalanobis_dist'], 1)
            poly = np.poly1d(z)
            x_range = np.linspace(avg['gss_polarization'].min(), avg['gss_polarization'].max(), 100)
            ax.plot(x_range, poly(x_range), 'k--', alpha=0.4)
            ax.text(0.5, 0.02, f'r={r:.3f}', transform=ax.transAxes,
                   ha='center', fontsize=9)

        ax.set_xlabel('GSS Polarization')
        ax.set_ylabel('Model Mahalanobis Distance')
        ax.set_title(f'{model_type.title()} Models')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'quadrant_analysis')
    plt.close()

    # Plot 2: False polarization rate by model type
    fp_rates = []
    for model_type in ['base', 'instruct', 'reasoning']:
        for family in df_valid['family'].unique():
            df_sub = df_valid[(df_valid['model_type'] == model_type) &
                              (df_valid['family'] == family)]
            if len(df_sub) == 0:
                continue

            low_gss = df_sub[df_sub['gss_high'] == 0]
            dist_median = df_sub['mahalanobis_dist'].median()
            fp = (low_gss['mahalanobis_dist'] > dist_median).mean() if len(low_gss) > 0 else 0

            fp_rates.append({
                'model_type': model_type,
                'family': family,
                'false_positive_rate': fp,
            })

    df_fp = pd.DataFrame(fp_rates)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_fp, x='model_type', y='false_positive_rate', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('False Polarization Rate by Model Type')
    ax.set_ylabel('False Positive Rate')
    ax.set_xlabel('Model Type')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'false_polarization_rate')
    plt.close()

    # Plot 3: ROC curves
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_type in ['base', 'instruct', 'reasoning']:
        df_mt = df_valid[df_valid['model_type'] == model_type].dropna(subset=['gss_high'])
        if len(df_mt) < 10:
            continue

        fpr, tpr, _ = roc_curve(df_mt['gss_high'], df_mt['mahalanobis_dist'])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model_type} (AUC={roc_auc_val:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC: Can Model Distance Predict Real Polarization?')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'roc_curves')
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 6 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_experiment()
