"""
Experiment 12: DW-NOMINATE Per-Head Linear Probing

Directly replicates and extends Kaplan et al. (ICLR 2025):
"Linear Representations of Political Perspective Emerge in Large Language Models"

Research Question:
    Do LLM attention heads linearly encode continuous political ideology
    (DW-NOMINATE scores)? Does this differ between base/instruct/reasoning models?

Background:
    Kaplan et al. trained ridge regression on individual attention heads to predict
    DW-NOMINATE scores, finding Spearman rho ~0.86 in middle-layer heads of
    instruct models (Llama-2-7b-chat, Mistral-7b-instruct, Vicuna-7b).
    They only tested instruct models and used a single prompt template.
    We extend this to base and reasoning models, multiple topics, and compare
    with our binary party classification results (Exp7).

Hypotheses:
    H12a: Instruct models have higher per-head DW-NOMINATE correlation than base
           (consistent with Exp1 binary result)
    H12b: The most predictive heads are concentrated in middle layers for instruct
           (replicating Kaplan et al.), but distributed differently for base/reasoning
    H12c: Per-head DW-NOMINATE probing shows CROSS-TOPIC transfer
           (unlike binary party classification in Exp3, because continuous
           ideology captures more than party identity)
    H12d: Reasoning models show lower DW-NOMINATE correlation than instruct
           (consistent with Exp1)

Method:
    - For each model x topic: extract (N, L, H, D) activations
    - For each head (l, h): train ridge regression to predict DW-NOMINATE dim1
    - Evaluate with 2-fold CV (matching Kaplan et al.)
    - Also test cross-topic transfer: train on topic A, predict on topic B
    - Compare with Kaplan et al.'s results on instruct models

Analysis:
    - Per-head Spearman correlation heatmaps (L x H)
    - Best-head correlation by model type
    - Cross-topic transfer of continuous probes
    - Layer-wise profile of ideology encoding
    - Comparison with binary party classification (Exp7)

Runtime: ~3 hours on H100 (8 models x 10 topics x per-head ridge regression)
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
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

EXPERIMENT_NAME = "exp12"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/data/HS116_members_fullname.csv"
N_TOPICS = 10  # Use 10 diverse topics for probing
RIDGE_ALPHA = 1.0  # Match Kaplan et al.
N_FOLDS = 2  # Match Kaplan et al.'s 2-fold CV
TOP_K_HEADS = 32  # Number of top heads for ensembling (match Kaplan et al.)

BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}
INSTRUCT_TEMPLATE_KEYS = {
    "public_issues": "default",
    "private_life":  "opinion",
}

# Kaplan et al. prompt template for direct comparison
KAPLAN_TEMPLATE = "Generate a statement by {name}, a politician in the United States."


# =============================================================================
# Per-Head Ridge Regression Probing
# =============================================================================

def compute_head_nominate_probing(
    activations: np.ndarray,
    nominate_scores: np.ndarray,
    alpha: float = 1.0,
    n_folds: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each attention head, train ridge regression to predict DW-NOMINATE scores.
    Uses 2-fold cross-validation (matching Kaplan et al.).

    Args:
        activations: (N, L, H, D) array
        nominate_scores: (N,) continuous DW-NOMINATE dim1 scores
        alpha: Ridge regression regularization
        n_folds: Number of CV folds

    Returns:
        spearman_matrix: (L, H) Spearman correlation per head
        pearson_matrix: (L, H) Pearson correlation per head
    """
    N, L, H, D = activations.shape
    spearman_matrix = np.zeros((L, H))
    pearson_matrix = np.zeros((L, H))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for l in range(L):
        for h in range(H):
            head_acts = activations[:, l, h, :]  # (N, D)

            # Collect cross-validated predictions
            all_preds = np.zeros(N)
            all_true = np.zeros(N)
            all_idx = np.zeros(N, dtype=int)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(head_acts)):
                X_train = head_acts[train_idx]
                y_train = nominate_scores[train_idx]
                X_test = head_acts[test_idx]
                y_test = nominate_scores[test_idx]

                # Standardize per fold
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Ridge regression
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train, y_train)
                preds = ridge.predict(X_test)

                all_preds[test_idx] = preds
                all_true[test_idx] = y_test

            # Compute correlations on pooled CV predictions
            try:
                rho, _ = spearmanr(all_true, all_preds)
                r, _ = pearsonr(all_true, all_preds)
                spearman_matrix[l, h] = rho if not np.isnan(rho) else 0.0
                pearson_matrix[l, h] = r if not np.isnan(r) else 0.0
            except Exception:
                spearman_matrix[l, h] = 0.0
                pearson_matrix[l, h] = 0.0

    return spearman_matrix, pearson_matrix


def compute_cross_topic_transfer(
    activations_train: np.ndarray,
    nominate_train: np.ndarray,
    activations_test: np.ndarray,
    nominate_test: np.ndarray,
    top_heads: List[Tuple[int, int]],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Train ridge regression on one topic's activations, test on another.
    Uses ensemble of top heads (matching Kaplan et al.).

    Args:
        activations_train: (N, L, H, D) from training topic
        nominate_train: (N,) DW-NOMINATE scores for training topic
        activations_test: (N, L, H, D) from test topic
        nominate_test: (N,) DW-NOMINATE scores for test topic
        top_heads: List of (layer, head) tuples to ensemble
        alpha: Ridge regularization

    Returns:
        Dict with spearman_rho, pearson_r for ensembled predictions
    """
    N_train = activations_train.shape[0]
    N_test = activations_test.shape[0]

    # Ensemble predictions from top heads
    ensemble_preds = np.zeros(N_test)

    for (l, h) in top_heads:
        X_train = activations_train[:, l, h, :]
        X_test = activations_test[:, l, h, :]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, nominate_train)
        preds = ridge.predict(X_test)

        ensemble_preds += preds

    ensemble_preds /= len(top_heads)

    try:
        rho, p_rho = spearmanr(nominate_test, ensemble_preds)
        r, p_r = pearsonr(nominate_test, ensemble_preds)
    except Exception:
        rho, p_rho = 0.0, 1.0
        r, p_r = 0.0, 1.0

    return {
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'pearson_r': r,
        'pearson_p': p_r,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 12: DW-NOMINATE Per-Head Probing."""

    print("=" * 80)
    print("EXPERIMENT 12: DW-NOMINATE PER-HEAD LINEAR PROBING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Replicating & extending Kaplan et al. (ICLR 2025)")
    print()

    set_random_seeds(42)

    # =========================================================================
    # Load data
    # =========================================================================

    # Load politicians with DW-NOMINATE scores
    print("Loading politicians with DW-NOMINATE scores...")
    df_politicians = pd.read_csv(POLITICIAN_CSV)

    # Filter to D+R with valid nominate scores
    df_dr = df_politicians[
        df_politicians['party_code'].isin([100, 200]) &
        df_politicians['nominate_dim1'].notna()
    ].copy()

    politician_names = df_dr['fullname'].tolist()
    party_labels = (df_dr['party_code'].values == 200).astype(int)
    nominate_dim1 = df_dr['nominate_dim1'].values.astype(np.float32)
    nominate_dim2 = df_dr['nominate_dim2'].values.astype(np.float32)

    n_dem = int(np.sum(party_labels == 0))
    n_rep = int(np.sum(party_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")
    print(f"DW-NOMINATE dim1: [{nominate_dim1.min():.3f}, {nominate_dim1.max():.3f}]")
    print(f"  Democrat mean: {nominate_dim1[party_labels == 0].mean():.3f}")
    print(f"  Republican mean: {nominate_dim1[party_labels == 1].mean():.3f}")

    # Load topics - use a mix of public and private
    print("\nLoading topics...")
    topic_list_path = TOPIC_LISTS_DIR / "exp12_nominate_topics.json"
    if topic_list_path.exists():
        with open(topic_list_path) as f:
            topics = json.load(f)
    else:
        # Create a diverse topic list
        from shared_utils import load_all_topics, sample_topics, get_excluded_topics
        public_topics, private_topics = load_all_topics()
        sampled_public, sampled_private = sample_topics(
            public_topics, private_topics,
            n_public=6, n_private=4,
            seed=12,  # Different seed from other experiments
            exclude=get_excluded_topics()
        )
        topics = {}
        for k, v in sampled_public.items():
            topics[k] = {"description": v, "category": "public_issues"}
        for k, v in sampled_private.items():
            topics[k] = {"description": v, "category": "private_life"}
        # Save for reproducibility
        with open(topic_list_path, 'w') as f:
            json.dump(topics, f, indent=2)
        print(f"  Created and saved topic list: {topic_list_path}")

    print(f"Using {len(topics)} topics for probing")
    for name, info in topics.items():
        cat = info['category'] if isinstance(info, dict) else 'public_issues'
        print(f"  - {name} ({cat})")

    # Also add a "Kaplan-style" no-topic prompt for direct comparison
    use_kaplan_prompt = True

    # =========================================================================
    # Run all models
    # =========================================================================

    all_results = []
    all_transfer_results = []
    model_topic_features = {}  # model_name -> topic_name -> (N, L, H, D)

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

            topic_spearman_matrices = {}
            topic_activations = {}  # Store for cross-topic transfer

            # -----------------------------------------------------------------
            # Kaplan-style prompt (no topic, for direct comparison)
            # -----------------------------------------------------------------
            if use_kaplan_prompt:
                print(f"  [Kaplan-style] No-topic prompt")

                if model_type == "base":
                    # Completion-style Kaplan prompt
                    prompts = [
                        f"In 2019, {name} said that"
                        for name in politician_names
                    ]
                    system_msg = ""
                else:
                    prompts = [
                        KAPLAN_TEMPLATE.format(name=name)
                        for name in politician_names
                    ]
                    system_msg = SYSTEM_MSG_POLITICIAN

                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=batch_size, max_length=MAX_LENGTH
                )
                if hasattr(activations, 'cpu'):
                    activations = activations.cpu().numpy()
                elif not isinstance(activations, np.ndarray):
                    activations = np.array(activations)

                spearman_mat, pearson_mat = compute_head_nominate_probing(
                    activations, nominate_dim1, alpha=RIDGE_ALPHA, n_folds=N_FOLDS
                )

                best_spearman = np.max(np.abs(spearman_mat))
                best_loc = np.unravel_index(np.argmax(np.abs(spearman_mat)), spearman_mat.shape)
                mean_spearman = np.mean(np.abs(spearman_mat))

                print(f"    Best head: layer={best_loc[0]}, head={best_loc[1]}, "
                      f"rho={spearman_mat[best_loc]:.4f}")
                print(f"    Mean |rho|: {mean_spearman:.4f}")

                all_results.append({
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': '_kaplan_notopic',
                    'category': 'kaplan',
                    'best_spearman': best_spearman,
                    'best_head_layer': int(best_loc[0]),
                    'best_head_idx': int(best_loc[1]),
                    'best_head_layer_frac': float(best_loc[0]) / n_layers,
                    'mean_abs_spearman': mean_spearman,
                    'median_abs_spearman': float(np.median(np.abs(spearman_mat))),
                    'n_heads_above_05': int(np.sum(np.abs(spearman_mat) > 0.5)),
                    'n_heads_above_07': int(np.sum(np.abs(spearman_mat) > 0.7)),
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                })

                topic_spearman_matrices['_kaplan_notopic'] = spearman_mat

                # Store activations for cross-topic transfer
                topic_activations['_kaplan_notopic'] = activations.copy()

                del activations
                gc.collect()
                torch.cuda.empty_cache()

            # -----------------------------------------------------------------
            # Per-topic probing
            # -----------------------------------------------------------------
            for topic_idx, (topic_name, topic_info) in enumerate(topics.items(), 1):
                if isinstance(topic_info, dict):
                    topic_desc = topic_info.get('description', topic_info.get('desc', topic_name))
                    category = topic_info.get('category', 'public_issues')
                else:
                    topic_desc = topic_info
                    category = 'public_issues'

                print(f"  [{topic_idx}/{len(topics)}] {topic_name} ({category})")

                # Generate prompts
                if model_type == "base":
                    template = BASE_TEMPLATES[category]
                    prompts = [
                        template.format(name=name, topic=topic_desc)
                        for name in politician_names
                    ]
                    system_msg = ""
                else:
                    key = INSTRUCT_TEMPLATE_KEYS[category]
                    template_str = POLITICIAN_TEMPLATES[key]
                    prompts = [
                        template_str.format(name=name, topic=topic_desc)
                        for name in politician_names
                    ]
                    system_msg = SYSTEM_MSG_POLITICIAN

                # Extract activations
                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=batch_size, max_length=MAX_LENGTH
                )
                if hasattr(activations, 'cpu'):
                    activations = activations.cpu().numpy()
                elif not isinstance(activations, np.ndarray):
                    activations = np.array(activations)

                # Per-head ridge regression probing
                spearman_mat, pearson_mat = compute_head_nominate_probing(
                    activations, nominate_dim1, alpha=RIDGE_ALPHA, n_folds=N_FOLDS
                )

                best_spearman = np.max(np.abs(spearman_mat))
                best_loc = np.unravel_index(
                    np.argmax(np.abs(spearman_mat)), spearman_mat.shape
                )
                mean_spearman = np.mean(np.abs(spearman_mat))

                print(f"    Best head: layer={best_loc[0]}, head={best_loc[1]}, "
                      f"rho={spearman_mat[best_loc]:.4f}")
                print(f"    Mean |rho|: {mean_spearman:.4f}")

                all_results.append({
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': topic_name,
                    'category': category,
                    'best_spearman': best_spearman,
                    'best_head_layer': int(best_loc[0]),
                    'best_head_idx': int(best_loc[1]),
                    'best_head_layer_frac': float(best_loc[0]) / n_layers,
                    'mean_abs_spearman': mean_spearman,
                    'median_abs_spearman': float(np.median(np.abs(spearman_mat))),
                    'n_heads_above_05': int(np.sum(np.abs(spearman_mat) > 0.5)),
                    'n_heads_above_07': int(np.sum(np.abs(spearman_mat) > 0.7)),
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                })

                topic_spearman_matrices[topic_name] = spearman_mat

                # Store activations for cross-topic transfer (keep only last 2 topics
                # to avoid OOM)
                if len(topic_activations) >= 3:
                    # Remove oldest non-kaplan topic
                    oldest = [k for k in topic_activations if k != '_kaplan_notopic'][0]
                    del topic_activations[oldest]
                topic_activations[topic_name] = activations.copy()

                del activations
                gc.collect()
                torch.cuda.empty_cache()

            # -----------------------------------------------------------------
            # Cross-topic transfer (using top heads from Kaplan prompt)
            # -----------------------------------------------------------------
            print(f"\n  Computing cross-topic transfer...")

            # Find top-K heads from Kaplan prompt
            if '_kaplan_notopic' in topic_spearman_matrices:
                kaplan_mat = topic_spearman_matrices['_kaplan_notopic']
            else:
                # Use average across topics
                kaplan_mat = np.mean(
                    [np.abs(m) for m in topic_spearman_matrices.values()], axis=0
                )

            flat_idx = np.argsort(np.abs(kaplan_mat).ravel())[::-1][:TOP_K_HEADS]
            top_heads = [
                (idx // kaplan_mat.shape[1], idx % kaplan_mat.shape[1])
                for idx in flat_idx
            ]

            # Transfer between available stored topic pairs
            stored_topics = list(topic_activations.keys())
            for i, t1 in enumerate(stored_topics):
                for j, t2 in enumerate(stored_topics):
                    if i >= j:
                        continue

                    transfer = compute_cross_topic_transfer(
                        topic_activations[t1], nominate_dim1,
                        topic_activations[t2], nominate_dim1,
                        top_heads, alpha=RIDGE_ALPHA
                    )

                    all_transfer_results.append({
                        'model_name': model_name,
                        'model_type': model_type,
                        'family': family_name,
                        'train_topic': t1,
                        'test_topic': t2,
                        'spearman_rho': transfer['spearman_rho'],
                        'pearson_r': transfer['pearson_r'],
                    })

                    # Also reverse direction
                    transfer_rev = compute_cross_topic_transfer(
                        topic_activations[t2], nominate_dim1,
                        topic_activations[t1], nominate_dim1,
                        top_heads, alpha=RIDGE_ALPHA
                    )
                    all_transfer_results.append({
                        'model_name': model_name,
                        'model_type': model_type,
                        'family': family_name,
                        'train_topic': t2,
                        'test_topic': t1,
                        'spearman_rho': transfer_rev['spearman_rho'],
                        'pearson_r': transfer_rev['pearson_r'],
                    })

            # Save per-model checkpoint
            save_checkpoint(
                {'results': all_results, 'transfer': all_transfer_results},
                EXPERIMENT_NAME, f"{model_name}_checkpoint"
            )

            # Cleanup
            del topic_activations, topic_spearman_matrices
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  Unloaded {model_name}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    df_results = pd.DataFrame(all_results)
    df_transfer = pd.DataFrame(all_transfer_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_probing_{timestamp}.csv"
    transfer_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_transfer_{timestamp}.csv"
    df_results.to_csv(results_path, index=False)
    df_transfer.to_csv(transfer_path, index=False)
    print(f"\nSaved results: {results_path}")
    print(f"Saved transfer: {transfer_path}")

    # --- Best-head Spearman by model type ---
    print("\n--- Best-Head Spearman Correlation by Model Type ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df_results[df_results['model_type'] == mt]
        if len(sub) == 0:
            continue
        print(f"  {mt}:")
        print(f"    Best |rho|: {sub['best_spearman'].mean():.4f} ± {sub['best_spearman'].std():.4f}")
        print(f"    Mean |rho|: {sub['mean_abs_spearman'].mean():.4f} ± {sub['mean_abs_spearman'].std():.4f}")
        print(f"    Heads > 0.5: {sub['n_heads_above_05'].mean():.1f}")
        print(f"    Heads > 0.7: {sub['n_heads_above_07'].mean():.1f}")
        print(f"    Best head layer frac: {sub['best_head_layer_frac'].mean():.3f}")

    # --- Kaplan comparison (no-topic prompts only) ---
    print("\n--- Kaplan et al. Comparison (no-topic prompts) ---")
    kaplan_df = df_results[df_results['topic_name'] == '_kaplan_notopic']
    if len(kaplan_df) > 0:
        print("  Kaplan et al. reference: best head rho ~0.85, middle layers (15-16)")
        for _, row in kaplan_df.iterrows():
            print(f"  {row['model_name']}: best rho={row['best_spearman']:.4f}, "
                  f"layer={row['best_head_layer']} ({row['best_head_layer_frac']:.2f}), "
                  f"heads>0.7={row['n_heads_above_07']}")

    # --- Cross-topic transfer ---
    if len(df_transfer) > 0:
        print("\n--- Cross-Topic Transfer (DW-NOMINATE probes) ---")
        for mt in ['base', 'instruct', 'reasoning']:
            sub = df_transfer[df_transfer['model_type'] == mt]
            if len(sub) == 0:
                continue
            print(f"  {mt}: mean transfer rho={sub['spearman_rho'].mean():.4f} ± {sub['spearman_rho'].std():.4f}")

    # --- Statistical tests ---
    from scipy.stats import ttest_ind
    print("\n--- Statistical Tests (best_spearman) ---")
    for mt1, mt2 in [('base', 'instruct'), ('base', 'reasoning'), ('instruct', 'reasoning')]:
        g1 = df_results[df_results['model_type'] == mt1]['best_spearman']
        g2 = df_results[df_results['model_type'] == mt2]['best_spearman']
        if len(g1) > 0 and len(g2) > 0:
            t, p = ttest_ind(g1, g2)
            print(f"  {mt1} vs {mt2}: t={t:.3f}, p={p:.4e}")

    # --- Plots ---
    setup_plot_style()

    # Bar chart: best Spearman by model type
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = df_results.groupby('model_type')['best_spearman'].agg(['mean', 'std']).reindex(['base', 'instruct', 'reasoning'])
    colors = {'base': '#2196F3', 'instruct': '#FF5722', 'reasoning': '#4CAF50'}
    bars = ax.bar(plot_data.index, plot_data['mean'], yerr=plot_data['std'],
                  color=[colors.get(x, 'gray') for x in plot_data.index],
                  capsize=5, alpha=0.8)
    ax.set_ylabel('Best Head Spearman |ρ|')
    ax.set_title('DW-NOMINATE Probing: Best Head Correlation by Model Type')
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Kaplan et al. reference')
    ax.legend()
    save_figure(fig, EXPERIMENT_NAME, f"best_spearman_{timestamp}")

    # Layer profile: mean |rho| by layer fraction
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax_idx, mt in enumerate(['base', 'instruct', 'reasoning']):
        ax = axes[ax_idx]
        sub = df_results[df_results['model_type'] == mt]
        if len(sub) == 0:
            continue
        ax.hist(sub['best_head_layer_frac'], bins=20, alpha=0.7,
                color=colors.get(mt, 'gray'), density=True)
        ax.set_xlabel('Best Head Layer (fraction of depth)')
        ax.set_title(f'{mt.capitalize()} Models')
        ax.axvline(x=sub['best_head_layer_frac'].mean(), color='black',
                    linestyle='--', label=f'mean={sub["best_head_layer_frac"].mean():.2f}')
        ax.legend()
    axes[0].set_ylabel('Density')
    fig.suptitle('Distribution of Best Head Layer Position')
    fig.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, f"layer_distribution_{timestamp}")

    # Transfer comparison
    if len(df_transfer) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        transfer_summary = df_transfer.groupby('model_type')['spearman_rho'].agg(['mean', 'std']).reindex(['base', 'instruct', 'reasoning'])
        bars = ax.bar(transfer_summary.index, transfer_summary['mean'],
                      yerr=transfer_summary['std'],
                      color=[colors.get(x, 'gray') for x in transfer_summary.index],
                      capsize=5, alpha=0.8)
        ax.set_ylabel('Cross-Topic Transfer Spearman ρ')
        ax.set_title('DW-NOMINATE Probe: Cross-Topic Transfer')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5,
                    label='Kaplan et al. news transfer (~0.80)')
        ax.legend()
        save_figure(fig, EXPERIMENT_NAME, f"transfer_{timestamp}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 12 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_experiment()
