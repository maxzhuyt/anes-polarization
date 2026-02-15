"""
Experiment 7: Head-Level Discriminability Analysis

Research Question:
    Which attention heads are most important for encoding partisan information?
    Do base/instruct/reasoning models use different heads?

Hypotheses:
    H7a: Partisan info is concentrated in a small subset of heads (<10%)
    H7b: Instruct models use later-layer heads more than base models
    H7c: The most discriminative heads overlap across topics (stable "political" heads)

Method:
    - For each model × topic: extract (N, L, H, D) activations
    - For each individual head (l, h): train logistic regression on (N, D) to predict party
    - Build (L, H) discriminability matrix (AUC scores)
    - Compare most discriminative heads across model types and topics

Analysis:
    - Identify top-K heads per model (AUC > threshold)
    - Compute overlap of top heads across topics (Jaccard similarity)
    - Compare head importance profiles across base/instruct/reasoning
    - Visualize as heatmaps

Runtime: ~2 hours on H100
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
POLITICIAN_CSV = "/project/jevans/maxzhuyt/data/HS116_members_fullname.csv"

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
    Compute discriminability for each individual attention head using a fast
    two-stage approach: LDA projection to 1D + ROC-AUC on the projection.

    This is much faster than per-head logistic regression with CV because:
    - LDA projection is closed-form (no iterative optimization)
    - AUC on 1D projections is O(N log N)
    - Total: O(L * H * N * D) instead of O(L * H * N * D * max_iter * n_folds)

    Args:
        activations: (N, L, H, D) array
        party_labels: (N,) party labels

    Returns:
        (L, H) matrix of AUC scores
    """
    assert activations.ndim == 4
    N, L, H, D = activations.shape

    auc_matrix = np.zeros((L, H))

    # Precompute masks
    mask0 = party_labels == 0
    mask1 = party_labels == 1

    for l in range(L):
        for h in range(H):
            # Extract single head: (N, D)
            head_acts = activations[:, l, h, :]

            # Compute class means
            mean0 = head_acts[mask0].mean(axis=0)
            mean1 = head_acts[mask1].mean(axis=0)

            # Compute pooled within-class covariance (regularized)
            diff0 = head_acts[mask0] - mean0
            diff1 = head_acts[mask1] - mean1
            Sw = (diff0.T @ diff0 + diff1.T @ diff1) / (N - 2)
            Sw += np.eye(D) * 1e-6  # Regularization

            # LDA projection direction
            try:
                w = np.linalg.solve(Sw, mean1 - mean0)
            except np.linalg.LinAlgError:
                auc_matrix[l, h] = 0.5
                continue

            # Project data onto LDA axis
            projections = head_acts @ w  # (N,)

            # Compute AUC on 1D projections
            try:
                auc_val = roc_auc_score(party_labels, projections)
                # Ensure AUC >= 0.5 (flip if needed)
                auc_val = max(auc_val, 1 - auc_val)
            except Exception:
                auc_val = 0.5

            auc_matrix[l, h] = auc_val

    return auc_matrix

def find_top_heads(
    auc_matrix: np.ndarray,
    threshold: float = 0.6,
    top_k: int = 20,
) -> List[Tuple[int, int, float]]:
    """
    Find the most discriminative heads.

    Returns list of (layer, head, auc) tuples sorted by AUC descending.
    """
    L, H = auc_matrix.shape
    heads = []
    for l in range(L):
        for h in range(H):
            if auc_matrix[l, h] > threshold:
                heads.append((l, h, float(auc_matrix[l, h])))

    heads.sort(key=lambda x: x[2], reverse=True)
    return heads[:top_k]

def compute_head_overlap(
    matrices: Dict[str, np.ndarray],
    threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Compute Jaccard overlap of top heads across topics/models.

    Args:
        matrices: Dict mapping name -> (L, H) AUC matrix
        threshold: AUC threshold for "important" heads

    Returns:
        DataFrame with pairwise Jaccard similarities
    """
    names = list(matrices.keys())
    n = len(names)
    overlap_matrix = np.zeros((n, n))

    # Get sets of important heads for each matrix
    head_sets = {}
    for name, mat in matrices.items():
        important = set()
        L, H = mat.shape
        for l in range(L):
            for h in range(H):
                if mat[l, h] > threshold:
                    important.add((l, h))
        head_sets[name] = important

    for i in range(n):
        for j in range(n):
            set_i = head_sets[names[i]]
            set_j = head_sets[names[j]]
            if len(set_i | set_j) > 0:
                overlap_matrix[i, j] = len(set_i & set_j) / len(set_i | set_j)
            else:
                overlap_matrix[i, j] = 0

    return pd.DataFrame(overlap_matrix, index=names, columns=names)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 7: Head-Level Discriminability."""

    print("="*80)
    print("EXPERIMENT 7: HEAD-LEVEL DISCRIMINABILITY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics (subset for efficiency)
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

    # GSS data
    gss_df = load_polarization_data()

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

            # Load model
            model, tokenizer = load_model(model_path)
            if model_type == "base":
                tokenizer.chat_template = None

            n_layers = model.config.num_hidden_layers
            n_heads = model.config.num_attention_heads

            topic_matrices = {}
            category = "public_issues"

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

                # Compute per-head AUC
                print(f"    Computing {n_layers}x{n_heads} head AUCs...")
                auc_matrix = compute_head_discriminability(activations, politician_labels)
                topic_matrices[topic_name] = auc_matrix

                # Find top heads
                top_heads = find_top_heads(auc_matrix, threshold=0.6, top_k=10)
                max_auc = float(np.max(auc_matrix))
                mean_auc = float(np.mean(auc_matrix))

                # Store result
                result = {
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': topic_name,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'max_auc': max_auc,
                    'mean_auc': mean_auc,
                    'n_heads_above_06': int(np.sum(auc_matrix > 0.6)),
                    'n_heads_above_07': int(np.sum(auc_matrix > 0.7)),
                    'top_head_layer': top_heads[0][0] if top_heads else -1,
                    'top_head_idx': top_heads[0][1] if top_heads else -1,
                    'top_head_auc': top_heads[0][2] if top_heads else 0.5,
                }
                all_results.append(result)

                print(f"    Max AUC: {max_auc:.4f}, "
                      f"Heads>0.6: {result['n_heads_above_06']}, "
                      f"Heads>0.7: {result['n_heads_above_07']}")

                torch.cuda.empty_cache()
                gc.collect()

            all_auc_matrices[model_name] = topic_matrices

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Per-model checkpoint
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

    # Summary
    print("\n--- Head Discriminability by Model Type ---")
    summary = df.groupby('model_type')[['max_auc', 'mean_auc', 'n_heads_above_06']].mean()
    print(summary.round(4))

    # Cross-topic overlap for each model
    print("\n--- Cross-Topic Head Overlap (Jaccard) ---")
    for model_name, topic_matrices in all_auc_matrices.items():
        if len(topic_matrices) > 1:
            overlap_df = compute_head_overlap(topic_matrices, threshold=0.6)
            # Mean off-diagonal overlap
            mask = ~np.eye(len(overlap_df), dtype=bool)
            mean_overlap = float(overlap_df.values[mask].mean())
            print(f"  {model_name}: mean cross-topic head overlap = {mean_overlap:.4f}")

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Heatmap of head AUCs for first topic, per model type
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

    # Plot 2: Summary - heads above threshold by model type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.boxplot(data=df, x='model_type', y='n_heads_above_06', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('Heads with AUC > 0.6')
    ax.set_ylabel('Number of Discriminative Heads')
    ax.set_xlabel('Model Type')

    ax = axes[1]
    sns.boxplot(data=df, x='model_type', y='max_auc', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('Max Head AUC')
    ax.set_ylabel('Best Single Head AUC')
    ax.set_xlabel('Model Type')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'discriminability_summary')
    plt.close()

    # Plot 3: Average AUC by layer (row-mean of heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, family_name in enumerate(MODEL_FAMILIES.keys()):
        ax = axes[idx]

        for model_type in ['base', 'instruct', 'reasoning']:
            model_name = f"{family_name}_{model_type}"
            if model_name not in all_auc_matrices:
                continue

            # Average AUC per layer across topics
            layer_aucs = []
            for topic_name, mat in all_auc_matrices[model_name].items():
                layer_mean = np.mean(mat, axis=1)  # Average across heads
                layer_aucs.append(layer_mean)

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
