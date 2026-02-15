"""
Experiment: Bramson et al. (2016) Nine Dimensions of Polarization

Tests which of the 9 independent polarization dimensions best distinguish
model-generated representations vs. actual GSS survey responses.

Nine Dimensions (Bramson et al. 2016):
1. Spread: Range or standard deviation of opinions
2. Dispersion: Variance across the distribution
3. Coverage: Proportion of opinion space occupied
4. Regionalization: Clustering/grouping of opinions
5. Fragmentation: Number of distinct peaks/modes
6. Distinctness: Separation between groups
7. Group Divergence: Distance between group means
8. Group Consensus: Within-group agreement
9. Size Parity: Balance in group sizes

Method:
- For each model × topic: extract activations, project to 1D (PC1)
- Compute all 9 dimensions on model distributions
- Compare to GSS response distributions on same topics
- Test which dimensions correlate with actual polarization

Hypotheses:
- H1: Models show higher group divergence than GSS (exaggerated separation)
- H2: Models show lower coverage than GSS (compressed opinion space)
- H3: Models show higher group consensus than GSS (less within-party variance)
- H4: Bimodality/fragmentation differs between base and instruct models

Runtime: ~4 hours (all models, subset of topics)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    set_random_seeds,
    load_polarization_data,
    save_checkpoint,
    setup_plot_style,
    save_figure,
    MODEL_FAMILIES,
    RESULTS_DIR,
    TOPIC_LISTS_DIR,
    compute_pca_and_distance,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from config import SYSTEM_MSG_POLITICIAN

# =============================================================================
# Bramson Dimension Calculations
# =============================================================================

def compute_spread(data: np.ndarray) -> float:
    """Dimension 1: Spread (range or std dev)."""
    return np.std(data, ddof=1)

def compute_dispersion(data: np.ndarray) -> float:
    """Dimension 2: Dispersion (variance)."""
    return np.var(data, ddof=1)

def compute_coverage(data: np.ndarray, opinion_space: Tuple[float, float] = None) -> float:
    """Dimension 3: Coverage (proportion of space occupied)."""
    if opinion_space is None:
        # Use standardized space [-3, 3] (3 std devs)
        opinion_space = (-3, 3)

    data_range = np.max(data) - np.min(data)
    space_range = opinion_space[1] - opinion_space[0]

    return data_range / space_range

def compute_regionalization(data: np.ndarray, n_regions: int = 5) -> float:
    """
    Dimension 4: Regionalization (clustering tendency).
    Use silhouette score with k-means.
    """
    if len(data) < 10:
        return 0.0

    # Reshape for sklearn
    X = data.reshape(-1, 1)

    # Try k-means with n_regions clusters
    try:
        kmeans = KMeans(n_clusters=min(n_regions, len(data)//2), random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        return max(0, score)  # Silhouette ranges [-1, 1], we want [0, 1]
    except:
        return 0.0

def compute_fragmentation(data: np.ndarray, min_prominence: float = 0.1) -> int:
    """
    Dimension 5: Fragmentation (number of modes/peaks).
    Use kernel density estimation and peak detection.
    """
    from scipy.stats import gaussian_kde

    if len(data) < 10:
        return 1

    # Create KDE
    kde = gaussian_kde(data, bw_method='scott')

    # Evaluate on grid
    x_grid = np.linspace(np.min(data) - 0.5, np.max(data) + 0.5, 200)
    density = kde(x_grid)

    # Find peaks
    peaks, properties = find_peaks(density, prominence=min_prominence * np.max(density))

    return max(1, len(peaks))

def compute_distinctness(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Dimension 6: Distinctness (separation between groups).
    Cohen's d effect size between two groups.
    """
    group0 = data[labels == 0]
    group1 = data[labels == 1]

    if len(group0) < 2 or len(group1) < 2:
        return 0.0

    mean_diff = np.abs(np.mean(group0) - np.mean(group1))
    pooled_std = np.sqrt((np.var(group0, ddof=1) + np.var(group1, ddof=1)) / 2)

    if pooled_std == 0:
        return 0.0

    return mean_diff / pooled_std

def compute_group_divergence(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Dimension 7: Group Divergence (between-group variance / total variance).
    Similar to eta-squared.
    """
    group0 = data[labels == 0]
    group1 = data[labels == 1]

    if len(group0) < 2 or len(group1) < 2:
        return 0.0

    overall_mean = np.mean(data)

    # Between-group variance
    ss_between = len(group0) * (np.mean(group0) - overall_mean)**2 + \
                 len(group1) * (np.mean(group1) - overall_mean)**2

    # Total variance
    ss_total = np.sum((data - overall_mean)**2)

    if ss_total == 0:
        return 0.0

    return ss_between / ss_total

def compute_group_consensus(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Dimension 8: Group Consensus (within-group homogeneity).
    Inverse of average within-group variance.
    """
    group0 = data[labels == 0]
    group1 = data[labels == 1]

    if len(group0) < 2 or len(group1) < 2:
        return 0.0

    var0 = np.var(group0, ddof=1)
    var1 = np.var(group1, ddof=1)

    avg_within_var = (var0 + var1) / 2

    # Return inverse (higher consensus = lower variance)
    if avg_within_var == 0:
        return 1.0

    return 1.0 / (1.0 + avg_within_var)

def compute_size_parity(labels: np.ndarray) -> float:
    """
    Dimension 9: Size Parity (balance between groups).
    1.0 = perfectly balanced, 0.0 = all in one group.
    """
    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)
    total = len(labels)

    if total == 0:
        return 0.0

    # Herfindahl index inverted
    p0 = n0 / total
    p1 = n1 / total

    # Perfect parity = 0.5, 0.5 -> H = 0.5, we want parity metric = 1
    # All in one group -> H = 1, we want parity metric = 0
    H = p0**2 + p1**2

    # Normalize: H ranges from 0.5 (perfect balance) to 1.0 (all one group)
    # Map to [1, 0]
    return 2 * (1 - H)

def compute_all_bramson_dimensions(data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute all 9 Bramson dimensions."""
    return {
        'spread': compute_spread(data),
        'dispersion': compute_dispersion(data),
        'coverage': compute_coverage(data),
        'regionalization': compute_regionalization(data),
        'fragmentation': compute_fragmentation(data),
        'distinctness': compute_distinctness(data, labels),
        'group_divergence': compute_group_divergence(data, labels),
        'group_consensus': compute_group_consensus(data, labels),
        'size_parity': compute_size_parity(labels),
    }

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Bramson dimensions experiment."""

    print("="*80)
    print("EXPERIMENT: BRAMSON ET AL. 9 DIMENSIONS OF POLARIZATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics - use Exp5 polarized topics (30 most polarized)
    print("Loading topics...")
    with open(TOPIC_LISTS_DIR / "exp5_polarized_topics.json") as f:
        public_topics = json.load(f)
    print(f"Loaded {len(public_topics)} polarized public issues")

    # Load politicians (same as run_model_comparison.py)
    print("\nLoading politicians...")
    politician_csv = "/project/jevans/maxzhuyt/data/HS116_members_fullname.csv"
    df_politicians = load_politicians(politician_csv)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)

    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Load GSS data for comparison
    print("\nLoading GSS polarization data...")
    gss_df = load_polarization_data()

    import gc
    import torch
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.preprocessing import StandardScaler

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

            for topic_idx, (topic_name, topic_desc) in enumerate(public_topics.items(), 1):
                print(f"  [{topic_idx}/{len(public_topics)}] {topic_name}")

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
                    batch_size=batch_size, max_length=128,
                )

                # Flatten and project to 1D immediately
                act_2d = activations.reshape(activations.shape[0], -1) if activations.ndim > 2 else activations
                scaler = StandardScaler()
                activations_scaled = scaler.fit_transform(act_2d)
                pca = SkPCA(n_components=1)
                data_1d = pca.fit_transform(activations_scaled).flatten()

                # Compute all Bramson dimensions
                dimensions = compute_all_bramson_dimensions(data_1d, politician_labels)

                result = {
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': topic_name,
                }
                result.update(dimensions)

                # Add GSS comparison
                gss_row = gss_df[gss_df['variable'] == topic_name]
                if len(gss_row) > 0:
                    result['gss_polarization'] = gss_row.iloc[0]['polarization']
                    result['gss_mean_diff'] = abs(gss_row.iloc[0]['mean_dem'] - gss_row.iloc[0]['mean_rep'])
                else:
                    result['gss_polarization'] = np.nan
                    result['gss_mean_diff'] = np.nan

                all_results.append(result)

                # Free memory
                del activations, act_2d, activations_scaled, data_1d
                torch.cuda.empty_cache()
                gc.collect()

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Save per-model checkpoint
            model_results = [r for r in all_results if r['model_name'] == model_name]
            checkpoint_path = save_checkpoint(
                model_results,
                'exp_bramson',
                model_name
            )
            print(f"\nSaved checkpoint: {checkpoint_path}")

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"bramson_dimensions_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results: {csv_path}")

    # Aggregate statistics
    dimension_cols = ['spread', 'dispersion', 'coverage', 'regionalization',
                     'fragmentation', 'distinctness', 'group_divergence',
                     'group_consensus', 'size_parity']

    print("\n--- Average Dimensions by Model Type ---")
    df_agg = df.groupby('model_type')[dimension_cols].mean()
    print(df_agg.round(3))

    # Correlation with GSS polarization
    print("\n--- Correlation with GSS Polarization ---")
    for dim in dimension_cols:
        corr = df[dim].corr(df['gss_polarization'])
        print(f"  {dim:20s}: r = {corr:.3f}")

    # Plots
    setup_plot_style()

    # Plot 1: All 9 dimensions by model type
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, dim in enumerate(dimension_cols):
        ax = axes[idx]

        df_plot = df.groupby(['model_type', 'model_name'])[dim].mean().reset_index()

        sns.boxplot(data=df_plot, x='model_type', y=dim, ax=ax,
                   order=['base', 'instruct', 'reasoning'])
        ax.set_title(f'{dim.replace("_", " ").title()}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_figure(fig, 'bramson', 'all_dimensions')
    plt.close()

    # Plot 2: Dimension × GSS polarization correlation
    fig, ax = plt.subplots(figsize=(10, 6))

    correlations = []
    for dim in dimension_cols:
        corr = df[dim].corr(df['gss_polarization'])
        correlations.append({'dimension': dim, 'correlation': corr})

    df_corr = pd.DataFrame(correlations)
    df_corr = df_corr.sort_values('correlation', ascending=False)

    sns.barplot(data=df_corr, x='correlation', y='dimension', ax=ax)
    ax.set_xlabel('Correlation with GSS Polarization')
    ax.set_ylabel('Bramson Dimension')
    ax.set_title('Which Dimensions Best Predict Actual Polarization?')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    save_figure(fig, 'bramson', 'gss_correlations')
    plt.close()

    # Plot 3: Model type comparison on key dimensions
    key_dims = ['group_divergence', 'group_consensus', 'distinctness', 'fragmentation']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, dim in enumerate(key_dims):
        ax = axes[idx]

        df_plot = df.groupby(['family', 'model_type'])[dim].mean().reset_index()

        sns.barplot(data=df_plot, x='family', y=dim, hue='model_type', ax=ax,
                   hue_order=['base', 'instruct', 'reasoning'])
        ax.set_title(f'{dim.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Model Type')

    plt.tight_layout()
    save_figure(fig, 'bramson', 'key_dimensions_by_family')
    plt.close()

    print(f"\n{'='*80}")
    print("BRAMSON DIMENSIONS EXPERIMENT COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_experiment()
