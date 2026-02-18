"""
Experiment 2: Layer-Depth Analysis

Research Question:
    At which transformer layers does partisan information emerge, peak,
    and potentially diminish? Do base/instruct/reasoning models differ?

Hypotheses:
    H2a: Partisan info peaks in middle layers for base models but late layers for instruct
    H2b: Reasoning models show earlier peak and faster decay (information "used up" for reasoning)
    H2c: Layer-profile similarity: instruct and reasoning share late-layer patterns

Method:
    - For each model × topic, compute Mahalanobis distance at EACH layer separately
    - activations[:, layer_idx, :, :] -> (N, H*D) -> PCA -> Mahalanobis
    - Compare layer profiles across model types
    - Find peak layer and FWHM (width) of partisan encoding

Analysis:
    - Plot Mahalanobis distance vs. layer for each model type
    - Identify peak layer, onset layer, and FWHM for each model
    - Compare profiles using Procrustes analysis or correlation
    - Test H2a-c with statistical tests

Runtime: ~3 hours on H100 (fewer topics but more compute per topic)
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
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

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from config import SYSTEM_MSG_POLITICIAN

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp2"
PCA_DIM = 15
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

# Prompt templates (same as shared_utils / run_model_comparison)
BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}
INSTRUCT_TEMPLATE_KEYS = {
    "public_issues": "default",
    "private_life":  "opinion",
}

# =============================================================================
# Per-Layer Analysis
# =============================================================================

def compute_per_layer_distances(
    activations: np.ndarray,
    party_labels: np.ndarray,
    pca_dim: int = 15,
) -> List[Dict]:
    """
    Compute Mahalanobis distance at each layer separately.

    Args:
        activations: (N, L, H, D) array
        party_labels: (N,) party labels
        pca_dim: PCA dimensionality

    Returns:
        List of dicts, one per layer, with layer_idx, mahalanobis_dist, variance_explained
    """
    assert activations.ndim == 4, f"Expected 4D activations, got {activations.ndim}D"
    N, L, H, D = activations.shape

    results = []
    for layer_idx in range(L):
        # Extract single layer: (N, H, D) -> flatten to (N, H*D)
        layer_acts = activations[:, layer_idx, :, :].reshape(N, -1)

        # PCA + Mahalanobis
        pca_result = compute_pca_and_distance(layer_acts, party_labels, pca_dim=pca_dim)

        results.append({
            'layer_idx': layer_idx,
            'layer_frac': layer_idx / (L - 1),  # 0.0 = first, 1.0 = last
            'mahalanobis_dist': pca_result['mahalanobis_dist'],
            'variance_explained': pca_result['variance_explained'],
        })

    return results

def find_peak_and_width(distances: np.ndarray) -> Dict:
    """
    Find the peak layer and FWHM of the partisan encoding profile.

    Args:
        distances: (L,) array of per-layer Mahalanobis distances

    Returns:
        Dict with peak_layer, peak_value, fwhm, onset_layer
    """
    peak_layer = int(np.argmax(distances))
    peak_value = float(distances[peak_layer])

    # FWHM: find where distance drops below half-max
    half_max = peak_value / 2
    above_half = distances >= half_max

    # Find first and last layer above half-max
    above_indices = np.where(above_half)[0]
    if len(above_indices) > 0:
        onset_layer = int(above_indices[0])
        offset_layer = int(above_indices[-1])
        fwhm = offset_layer - onset_layer + 1
    else:
        onset_layer = peak_layer
        fwhm = 1

    return {
        'peak_layer': peak_layer,
        'peak_value': peak_value,
        'fwhm': fwhm,
        'onset_layer': onset_layer,
        'peak_layer_frac': peak_layer / (len(distances) - 1) if len(distances) > 1 else 0,
    }

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 2: Layer-Depth Analysis."""

    print("="*80)
    print("EXPERIMENT 2: LAYER-DEPTH ANALYSIS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics
    print("Loading topics...")
    with open(TOPIC_LISTS_DIR / "exp2_public.json") as f:
        public_topics = json.load(f)
    with open(TOPIC_LISTS_DIR / "exp2_private.json") as f:
        private_topics = json.load(f)

    topics_by_category = {
        'public_issues': public_topics,
        'private_life': private_topics,
    }
    total_topics = sum(len(t) for t in topics_by_category.values())
    print(f"Loaded {total_topics} topics (public={len(public_topics)}, private={len(private_topics)})")

    # Load politicians
    print("\nLoading politicians...")
    df_politicians = load_politicians(POLITICIAN_CSV)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)
    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Load GSS data
    gss_df = load_polarization_data()

    # Results storage
    all_results = []       # Per-layer per-topic results
    profile_results = []   # Per-model per-topic peak/width summaries

    # Run all models
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

            # Load model
            model, tokenizer = load_model(model_path)
            if model_type == "base":
                tokenizer.chat_template = None

            n_layers = model.config.num_hidden_layers

            for category, topics in topics_by_category.items():
                print(f"\n  Category: {category}")

                for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                    print(f"  [{topic_idx}/{len(topics)}] {topic_name}")

                    # Generate prompts (same as shared_utils)
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

                    # Extract activations (N, L, H, D)
                    activations = extract_heads_batched(
                        model, tokenizer, prompts, system_msg,
                        batch_size=batch_size, max_length=MAX_LENGTH,
                    )

                    # Per-layer analysis
                    layer_results = compute_per_layer_distances(
                        activations, politician_labels, pca_dim=PCA_DIM
                    )

                    # Store per-layer results
                    for lr in layer_results:
                        lr.update({
                            'family': family_name,
                            'variant': variant_name,
                            'model_name': model_name,
                            'model_type': model_type,
                            'category': category,
                            'topic_name': topic_name,
                            'n_layers': n_layers,
                        })
                        all_results.append(lr)

                    # Compute peak/width profile
                    distances = np.array([lr['mahalanobis_dist'] for lr in layer_results])
                    profile = find_peak_and_width(distances)
                    profile.update({
                        'family': family_name,
                        'variant': variant_name,
                        'model_name': model_name,
                        'model_type': model_type,
                        'category': category,
                        'topic_name': topic_name,
                        'n_layers': n_layers,
                        'mean_distance': float(np.mean(distances)),
                        'max_distance': float(np.max(distances)),
                    })

                    # Add GSS polarization
                    gss_row = gss_df[gss_df['variable'] == topic_name]
                    if len(gss_row) > 0:
                        profile['gss_polarization'] = gss_row.iloc[0]['polarization']
                    else:
                        profile['gss_polarization'] = np.nan

                    profile_results.append(profile)

                    # Memory cleanup
                    torch.cuda.empty_cache()
                    gc.collect()

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Per-model checkpoint
            model_layer_results = [r for r in all_results if r['model_name'] == model_name]
            save_checkpoint(model_layer_results, EXPERIMENT_NAME, model_name)

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df_layers = pd.DataFrame(all_results)
    df_profiles = pd.DataFrame(profile_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_layers.to_csv(RESULTS_DIR / f"exp2_layers_{timestamp}.csv", index=False)
    df_profiles.to_csv(RESULTS_DIR / f"exp2_profiles_{timestamp}.csv", index=False)
    print(f"Saved layer results: {len(df_layers)} rows")
    print(f"Saved profile results: {len(df_profiles)} rows")

    # Summary: peak layer by model type
    print("\n--- Peak Layer (fraction) by Model Type ---")
    peak_summary = df_profiles.groupby('model_type')['peak_layer_frac'].agg(['mean', 'std']).round(3)
    print(peak_summary)

    print("\n--- FWHM by Model Type ---")
    fwhm_summary = df_profiles.groupby('model_type')['fwhm'].agg(['mean', 'std']).round(2)
    print(fwhm_summary)

    print("\n--- Peak Distance by Model Type ---")
    peak_dist_summary = df_profiles.groupby('model_type')['peak_value'].agg(['mean', 'std']).round(3)
    print(peak_dist_summary)

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Layer profile by model type (averaged across topics)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, family_name in enumerate(MODEL_FAMILIES.keys()):
        ax = axes[idx]
        df_family = df_layers[df_layers['family'] == family_name]

        for model_type in ['base', 'instruct', 'reasoning']:
            df_mt = df_family[df_family['model_type'] == model_type]
            if len(df_mt) == 0:
                continue

            # Average across topics, group by layer_frac
            avg_profile = df_mt.groupby('layer_frac')['mahalanobis_dist'].agg(['mean', 'std']).reset_index()

            ax.plot(avg_profile['layer_frac'], avg_profile['mean'], label=model_type, linewidth=2)
            ax.fill_between(
                avg_profile['layer_frac'],
                avg_profile['mean'] - avg_profile['std'],
                avg_profile['mean'] + avg_profile['std'],
                alpha=0.2
            )

        ax.set_title(f'{family_name}')
        ax.set_xlabel('Layer Position (fraction)')
        ax.set_ylabel('Mahalanobis Distance')
        ax.legend()

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'layer_profiles')
    plt.close()

    # Plot 2: Peak layer distribution by model type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.boxplot(data=df_profiles, x='model_type', y='peak_layer_frac', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('Peak Layer Position')
    ax.set_ylabel('Layer Fraction (0=first, 1=last)')
    ax.set_xlabel('Model Type')

    ax = axes[1]
    sns.boxplot(data=df_profiles, x='model_type', y='fwhm', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('FWHM of Partisan Encoding')
    ax.set_ylabel('Number of Layers (FWHM)')
    ax.set_xlabel('Model Type')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'peak_and_width')
    plt.close()

    # Plot 3: Layer profiles by category
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for cat_idx, category in enumerate(['public_issues', 'private_life']):
        for fam_idx, family_name in enumerate(MODEL_FAMILIES.keys()):
            ax = axes[cat_idx, fam_idx]
            df_sub = df_layers[(df_layers['family'] == family_name) &
                               (df_layers['category'] == category)]

            for model_type in ['base', 'instruct', 'reasoning']:
                df_mt = df_sub[df_sub['model_type'] == model_type]
                if len(df_mt) == 0:
                    continue

                avg_profile = df_mt.groupby('layer_frac')['mahalanobis_dist'].mean().reset_index()
                ax.plot(avg_profile['layer_frac'], avg_profile['mahalanobis_dist'],
                       label=model_type, linewidth=2)

            ax.set_title(f'{family_name} - {category.replace("_", " ").title()}')
            ax.set_xlabel('Layer Position')
            ax.set_ylabel('Mahalanobis Dist')
            ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'profiles_by_category')
    plt.close()

    # Plot 4: Heatmap of per-layer distances for each model
    unique_models = df_layers['model_name'].unique()
    n_models = len(unique_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 3 * n_models))
    if n_models == 1:
        axes = [axes]

    for m_idx, model_name in enumerate(unique_models):
        ax = axes[m_idx]
        df_m = df_layers[df_layers['model_name'] == model_name]

        # Pivot to get layer × topic matrix
        pivot = df_m.pivot_table(
            values='mahalanobis_dist',
            index='topic_name',
            columns='layer_idx',
            aggfunc='mean'
        )

        sns.heatmap(pivot, ax=ax, cmap='YlOrRd', xticklabels=5)
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'heatmaps')
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 2 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_experiment()
