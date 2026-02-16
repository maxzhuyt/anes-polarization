"""
Experiment 7: Head-Level Discriminability Analysis (Redesigned)

Research Question:
    Which attention heads are most important for encoding partisan information?
    Are the *same* heads consistently most discriminative across different topics?

Key Design Choice:
    Use top-k% heads (by AUC) instead of a fixed threshold (e.g., AUC > 0.6).
    A low threshold like 0.6 captures most heads, making Jaccard overlap trivially
    high. By taking only the top 5% or 10% most discriminative heads per topic,
    we test whether partisan encoding is truly concentrated in a small, stable
    set of specialized heads.

Method:
    - For each model × topic: extract (N, L, H, D) activations
    - For each individual head (l, h): LDA projection + AUC to measure discriminability
    - Take top-k% heads per topic (k = 5%, 10%)
    - Compute pairwise Jaccard overlap of top-k head sets across topics
    - Analyze which layers the top heads cluster in
    - Compare base vs instruct vs reasoning

Runtime: ~2 hours on H100/A100
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torch

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
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from config import SYSTEM_MSG_POLITICIAN

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp7"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

# Top-k percentages to analyze
TOP_K_PERCENTAGES = [0.05, 0.10]  # 5% and 10%

BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}
INSTRUCT_TEMPLATE_KEYS = {
    "public_issues": "default",
    "private_life":  "opinion",
}

# =============================================================================
# Head-Level Analysis
# =============================================================================

def compute_head_discriminability(
    activations: np.ndarray,
    party_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute discriminability for each individual attention head using
    LDA projection to 1D + ROC-AUC.

    Args:
        activations: (N, L, H, D) array
        party_labels: (N,) party labels

    Returns:
        (L, H) matrix of AUC scores
    """
    assert activations.ndim == 4
    N, L, H, D = activations.shape

    auc_matrix = np.zeros((L, H))

    mask0 = party_labels == 0
    mask1 = party_labels == 1

    for l in range(L):
        for h in range(H):
            head_acts = activations[:, l, h, :]

            mean0 = head_acts[mask0].mean(axis=0)
            mean1 = head_acts[mask1].mean(axis=0)

            diff0 = head_acts[mask0] - mean0
            diff1 = head_acts[mask1] - mean1
            Sw = (diff0.T @ diff0 + diff1.T @ diff1) / (N - 2)
            Sw += np.eye(D) * 1e-6

            try:
                w = np.linalg.solve(Sw, mean1 - mean0)
            except np.linalg.LinAlgError:
                auc_matrix[l, h] = 0.5
                continue

            projections = head_acts @ w

            try:
                auc_val = roc_auc_score(party_labels, projections)
                auc_val = max(auc_val, 1 - auc_val)
            except Exception:
                auc_val = 0.5

            auc_matrix[l, h] = auc_val

    return auc_matrix


def get_top_k_heads(
    auc_matrix: np.ndarray,
    top_fraction: float,
) -> set:
    """
    Get the top-k% most discriminative heads by AUC.

    Args:
        auc_matrix: (L, H) AUC matrix
        top_fraction: Fraction of heads to select (e.g., 0.05 for top 5%)

    Returns:
        Set of (layer, head) tuples
    """
    L, H = auc_matrix.shape
    total_heads = L * H
    k = max(1, int(total_heads * top_fraction))

    # Flatten, get top-k indices
    flat_indices = np.argsort(auc_matrix.flatten())[::-1][:k]
    top_heads = set()
    for idx in flat_indices:
        l = idx // H
        h = idx % H
        top_heads.add((l, h))
    return top_heads


def compute_pairwise_jaccard(
    head_sets: Dict[str, set],
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity between head sets.

    Returns:
        DataFrame with pairwise Jaccard values
    """
    names = list(head_sets.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            si = head_sets[names[i]]
            sj = head_sets[names[j]]
            union = len(si | sj)
            matrix[i, j] = len(si & sj) / union if union > 0 else 0.0

    return pd.DataFrame(matrix, index=names, columns=names)


def layer_distribution_of_top_heads(
    top_heads: set,
    n_layers: int,
) -> np.ndarray:
    """
    Count how many top heads fall in each layer.

    Returns:
        (n_layers,) array of counts
    """
    counts = np.zeros(n_layers, dtype=int)
    for (l, h) in top_heads:
        counts[l] += 1
    return counts


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 7: Top-K Head Discriminability."""

    print("="*80)
    print("EXPERIMENT 7: TOP-K HEAD DISCRIMINABILITY (REDESIGNED)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Top-k fractions: {TOP_K_PERCENTAGES}")
    print()

    set_random_seeds(42)

    # Load topics
    print("Loading topics...")
    with open(TOPIC_LISTS_DIR / "exp7_attention_topics.json") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} topics for head analysis")

    # Load politicians
    print("\nLoading politicians...")
    df_politicians = load_politicians(POLITICIAN_CSV)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)
    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Results storage
    all_results = []
    all_auc_matrices = {}  # model_name -> topic_name -> (L, H) matrix

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

            model, tokenizer = load_model(model_path)
            if model_type == "base":
                tokenizer.chat_template = None

            n_layers = model.config.num_hidden_layers
            n_heads = model.config.num_attention_heads
            total_heads = n_layers * n_heads

            topic_matrices = {}
            category = "public_issues"

            for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                print(f"  [{topic_idx}/{len(topics)}] {topic_name}")

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

                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=batch_size, max_length=MAX_LENGTH,
                )

                print(f"    Computing {n_layers}x{n_heads} head AUCs...")
                auc_matrix = compute_head_discriminability(activations, politician_labels)
                topic_matrices[topic_name] = auc_matrix

                # Per-topic stats
                max_auc = float(np.max(auc_matrix))
                mean_auc = float(np.mean(auc_matrix))
                median_auc = float(np.median(auc_matrix))

                for top_frac in TOP_K_PERCENTAGES:
                    top_heads = get_top_k_heads(auc_matrix, top_frac)
                    k = len(top_heads)
                    # Min AUC among top-k (the threshold this implies)
                    top_aucs = sorted([auc_matrix[l, h] for (l, h) in top_heads], reverse=True)
                    min_top_auc = top_aucs[-1] if top_aucs else 0.5

                    # Layer distribution
                    layer_counts = layer_distribution_of_top_heads(top_heads, n_layers)
                    top_layer = int(np.argmax(layer_counts))
                    top_half = "later" if top_layer >= n_layers // 2 else "earlier"

                    result = {
                        'family': family_name,
                        'variant': variant_name,
                        'model_name': model_name,
                        'model_type': model_type,
                        'topic_name': topic_name,
                        'n_layers': n_layers,
                        'n_heads_per_layer': n_heads,
                        'total_heads': total_heads,
                        'top_k_fraction': top_frac,
                        'top_k_count': k,
                        'max_auc': max_auc,
                        'mean_auc': mean_auc,
                        'median_auc': median_auc,
                        'min_top_k_auc': min_top_auc,
                        'top_layer_idx': top_layer,
                        'top_half': top_half,
                    }
                    all_results.append(result)

                print(f"    Max AUC: {max_auc:.4f}, Mean: {mean_auc:.4f}, "
                      f"Median: {median_auc:.4f}")

                del activations
                torch.cuda.empty_cache()
                gc.collect()

            all_auc_matrices[model_name] = topic_matrices

            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            save_checkpoint(
                {'results': [r for r in all_results if r['model_name'] == model_name],
                 'auc_matrices': topic_matrices},
                EXPERIMENT_NAME, model_name
            )

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"exp7_head_discriminability_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # --- AUC distribution summary ---
    print("\n--- AUC Distribution by Model Type ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df[(df['model_type'] == mt) & (df['top_k_fraction'] == TOP_K_PERCENTAGES[0])]
        if len(sub) == 0:
            continue
        print(f"  {mt}: max={sub['max_auc'].mean():.4f}, "
              f"mean={sub['mean_auc'].mean():.4f}, "
              f"median={sub['median_auc'].mean():.4f}")

    # --- Top-k cross-topic overlap ---
    print("\n--- Cross-Topic Top-K Head Overlap (Jaccard) ---")
    overlap_summary = []

    for model_name, topic_matrices in all_auc_matrices.items():
        if len(topic_matrices) < 2:
            continue

        for top_frac in TOP_K_PERCENTAGES:
            # Get top-k head sets per topic
            head_sets = {}
            for topic_name, mat in topic_matrices.items():
                head_sets[topic_name] = get_top_k_heads(mat, top_frac)

            # Compute pairwise Jaccard
            jaccard_df = compute_pairwise_jaccard(head_sets)
            mask = ~np.eye(len(jaccard_df), dtype=bool)
            mean_jaccard = float(jaccard_df.values[mask].mean())
            min_jaccard = float(jaccard_df.values[mask].min())
            max_jaccard = float(jaccard_df.values[mask].max())

            # Intersection of ALL topic top-k sets
            all_sets = list(head_sets.values())
            intersection = all_sets[0]
            for s in all_sets[1:]:
                intersection = intersection & s
            n_universal = len(intersection)

            model_type = model_name.split('_')[-1]
            family = '_'.join(model_name.split('_')[:-1])

            overlap_summary.append({
                'model_name': model_name,
                'family': family,
                'model_type': model_type,
                'top_k_fraction': top_frac,
                'mean_jaccard': mean_jaccard,
                'min_jaccard': min_jaccard,
                'max_jaccard': max_jaccard,
                'n_universal_heads': n_universal,
                'top_k_count': len(all_sets[0]),
            })

            pct_str = f"{int(top_frac*100)}%"
            print(f"  {model_name} (top {pct_str}): "
                  f"Jaccard mean={mean_jaccard:.4f}, "
                  f"range=[{min_jaccard:.4f}, {max_jaccard:.4f}], "
                  f"universal={n_universal}/{len(all_sets[0])}")

    df_overlap = pd.DataFrame(overlap_summary)
    overlap_path = RESULTS_DIR / f"exp7_overlap_summary_{timestamp}.csv"
    df_overlap.to_csv(overlap_path, index=False)
    print(f"\nSaved overlap summary: {overlap_path}")

    # --- Layer concentration ---
    print("\n--- Layer Concentration of Top Heads ---")
    for model_name, topic_matrices in all_auc_matrices.items():
        n_layers = list(topic_matrices.values())[0].shape[0]
        combined_counts = np.zeros(n_layers)

        for topic_name, mat in topic_matrices.items():
            top_heads = get_top_k_heads(mat, TOP_K_PERCENTAGES[0])
            combined_counts += layer_distribution_of_top_heads(top_heads, n_layers)

        # Normalize to fraction
        combined_counts /= combined_counts.sum()
        peak_layer = int(np.argmax(combined_counts))
        peak_frac = combined_counts[peak_layer]
        later_half_frac = combined_counts[n_layers//2:].sum()

        print(f"  {model_name}: peak layer={peak_layer}/{n_layers-1} "
              f"({peak_frac:.2%}), later half={later_half_frac:.2%}")

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: AUC heatmaps for first topic
    first_topic = list(topics.keys())[0]
    model_names_to_plot = [mn for mn in all_auc_matrices if first_topic in all_auc_matrices[mn]]

    if model_names_to_plot:
        n_plots = min(len(model_names_to_plot), 8)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for idx, model_name in enumerate(model_names_to_plot[:n_plots]):
            ax = axes[idx]
            mat = all_auc_matrices[model_name][first_topic]
            sns.heatmap(mat, ax=ax, cmap='YlOrRd', vmin=0.5, vmax=1.0,
                       xticklabels=8, yticklabels=4)
            ax.set_title(f'{model_name} - {first_topic}')
            ax.set_xlabel('Head')
            ax.set_ylabel('Layer')

        plt.tight_layout()
        save_figure(fig, EXPERIMENT_NAME, f'head_heatmaps_{first_topic}')
        plt.close()

    # Plot 2: Top-k overlap comparison across model types
    if len(df_overlap) > 0:
        fig, axes = plt.subplots(1, len(TOP_K_PERCENTAGES), figsize=(7*len(TOP_K_PERCENTAGES), 5))
        if len(TOP_K_PERCENTAGES) == 1:
            axes = [axes]

        for idx, top_frac in enumerate(TOP_K_PERCENTAGES):
            ax = axes[idx]
            sub = df_overlap[df_overlap['top_k_fraction'] == top_frac]
            if len(sub) == 0:
                continue

            sns.barplot(data=sub, x='model_type', y='mean_jaccard', ax=ax,
                       order=['base', 'instruct', 'reasoning'], ci=None)
            ax.set_title(f'Cross-Topic Head Overlap (Top {int(top_frac*100)}%)')
            ax.set_ylabel('Mean Pairwise Jaccard')
            ax.set_xlabel('Model Type')
            ax.set_ylim(0, 1)

        plt.tight_layout()
        save_figure(fig, EXPERIMENT_NAME, 'topk_overlap_by_type')
        plt.close()

    # Plot 3: Layer distribution of top heads
    fig, axes = plt.subplots(1, len(MODEL_FAMILIES), figsize=(6*len(MODEL_FAMILIES), 5))
    if len(MODEL_FAMILIES) == 1:
        axes = [axes]

    for idx, family_name in enumerate(MODEL_FAMILIES.keys()):
        ax = axes[idx]

        for model_type in ['base', 'instruct', 'reasoning']:
            model_name = f"{family_name}_{model_type}"
            if model_name not in all_auc_matrices:
                continue

            topic_matrices_m = all_auc_matrices[model_name]
            n_layers = list(topic_matrices_m.values())[0].shape[0]
            combined_counts = np.zeros(n_layers)

            for mat in topic_matrices_m.values():
                top_heads = get_top_k_heads(mat, TOP_K_PERCENTAGES[0])
                combined_counts += layer_distribution_of_top_heads(top_heads, n_layers)

            combined_counts /= combined_counts.sum()
            layer_fracs = np.arange(n_layers) / (n_layers - 1)
            ax.plot(layer_fracs, combined_counts, label=model_type, linewidth=2)

        ax.set_title(f'{family_name}')
        ax.set_xlabel('Layer Position (fraction)')
        ax.set_ylabel(f'Fraction of Top {int(TOP_K_PERCENTAGES[0]*100)}% Heads')
        ax.legend()

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'layer_distribution_top_heads')
    plt.close()

    # Plot 4: Mean AUC by layer (row-mean of heatmap)
    fig, axes = plt.subplots(1, len(MODEL_FAMILIES), figsize=(6*len(MODEL_FAMILIES), 5))
    if len(MODEL_FAMILIES) == 1:
        axes = [axes]

    for idx, family_name in enumerate(MODEL_FAMILIES.keys()):
        ax = axes[idx]

        for model_type in ['base', 'instruct', 'reasoning']:
            model_name = f"{family_name}_{model_type}"
            if model_name not in all_auc_matrices:
                continue

            layer_aucs = []
            for mat in all_auc_matrices[model_name].values():
                layer_aucs.append(np.mean(mat, axis=1))

            if layer_aucs:
                avg_layer = np.mean(layer_aucs, axis=0)
                n_layers = len(avg_layer)
                layer_fracs = np.arange(n_layers) / (n_layers - 1)
                ax.plot(layer_fracs, avg_layer, label=model_type, linewidth=2)

        ax.set_title(f'{family_name}')
        ax.set_xlabel('Layer Position (fraction)')
        ax.set_ylabel('Mean Head AUC')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'auc_by_layer')
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 7 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_experiment()
