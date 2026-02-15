"""
Experiment 3: Cross-Topic Representational Coherence

Research Question:
    Do models create a coherent political space, or are partisan representations
    topic-specific? Can a classifier trained on one topic predict party on another?

Hypotheses:
    H3a: Instruct models show higher cross-topic transfer than base (more coherent)
    H3b: Related topics (e.g., abortion variants) have higher transfer than unrelated
    H3c: Public issues show higher coherence than private life topics

Method:
    - For each model: extract activations on N core topics
    - For each pair of topics (i, j): train logistic regression on topic i, test on j
    - Build NxN transfer matrix
    - Compute coherence score = mean off-diagonal accuracy

Analysis:
    - Compare coherence scores across model types
    - Cluster topics by transfer similarity
    - Correlate coherence with GSS polarization

Runtime: ~3 hours on H100
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import ttest_ind
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

EXPERIMENT_NAME = "exp3"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}
INSTRUCT_TEMPLATE_KEYS = {
    "public_issues": "default",
    "private_life":  "opinion",
}

# =============================================================================
# Cross-Topic Transfer
# =============================================================================

def compute_transfer_matrix(
    features_by_topic: Dict[str, np.ndarray],
    party_labels: np.ndarray,
) -> Dict:
    """
    Compute cross-topic transfer matrix.

    For each pair (i, j), train logistic regression on topic i, test on topic j.

    Args:
        features_by_topic: Dict mapping topic_name -> (N, pca_dim) PCA-reduced features
        party_labels: (N,) party labels

    Returns:
        Dict with transfer_matrix, topic_names, accuracies, coherence_score
    """
    topic_names = list(features_by_topic.keys())
    n_topics = len(topic_names)

    # Compute transfer matrix
    transfer_matrix = np.zeros((n_topics, n_topics))
    auc_matrix = np.zeros((n_topics, n_topics))

    for i, train_topic in enumerate(topic_names):
        X_train = features_by_topic[train_topic]
        y_train = party_labels

        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, y_train)

        for j, test_topic in enumerate(topic_names):
            X_test = features_by_topic[test_topic]
            y_test = party_labels

            # Predict
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            transfer_matrix[i, j] = accuracy_score(y_test, y_pred)
            try:
                auc_matrix[i, j] = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc_matrix[i, j] = 0.5

    # Coherence = mean off-diagonal accuracy
    mask = ~np.eye(n_topics, dtype=bool)
    coherence_accuracy = float(np.mean(transfer_matrix[mask]))
    coherence_auc = float(np.mean(auc_matrix[mask]))

    # Diagonal = self-accuracy (sanity check)
    self_accuracy = float(np.mean(np.diag(transfer_matrix)))

    return {
        'transfer_matrix': transfer_matrix,
        'auc_matrix': auc_matrix,
        'topic_names': topic_names,
        'coherence_accuracy': coherence_accuracy,
        'coherence_auc': coherence_auc,
        'self_accuracy': self_accuracy,
    }

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 3: Cross-Topic Coherence."""

    print("="*80)
    print("EXPERIMENT 3: CROSS-TOPIC REPRESENTATIONAL COHERENCE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics
    print("Loading topics...")
    with open(TOPIC_LISTS_DIR / "exp3_core_topics.json") as f:
        all_topics = json.load(f)
    print(f"Loaded {len(all_topics)} core topics")

    # Split into categories based on topic prefix/source
    # For coherence, we use all topics together (category doesn't matter for transfer)
    # But we track it for analysis
    public_pol = load_polarization_data()
    public_vars = set(public_pol[public_pol['area'] == 'public_issues']['variable'].values) \
        if 'area' in public_pol.columns else set()

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

            # Extract activations and PCA-reduce immediately to save memory
            # Full 4D: 550 × 36 × 32 × 128 × 4 bytes = 325 MB/topic → OOM with 41 topics
            # PCA-reduced: 550 × 15 × 4 bytes = 33 KB/topic → negligible
            features_by_topic = {}
            category = "public_issues"
            pca_dim = 15

            for topic_idx, (topic_name, topic_desc) in enumerate(all_topics.items(), 1):
                print(f"  [{topic_idx}/{len(all_topics)}] {topic_name}")

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

                # PCA-reduce immediately: (N, L, H, D) -> (N, pca_dim)
                acts_2d = activations.reshape(activations.shape[0], -1) if activations.ndim > 2 else activations
                scaler = StandardScaler()
                acts_scaled = scaler.fit_transform(acts_2d)
                from sklearn.decomposition import PCA as SkPCA
                pca = SkPCA(n_components=min(pca_dim, acts_scaled.shape[1]))
                features_by_topic[topic_name] = pca.fit_transform(acts_scaled)

                # Free full activations immediately
                del activations, acts_2d, acts_scaled
                torch.cuda.empty_cache()
                gc.collect()

            # Unload model before heavy compute
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Compute transfer matrix using PCA-reduced features
            print(f"\n  Computing transfer matrix ({len(all_topics)}x{len(all_topics)})...")
            transfer_result = compute_transfer_matrix(
                features_by_topic, politician_labels,
            )

            print(f"  Coherence (accuracy): {transfer_result['coherence_accuracy']:.4f}")
            print(f"  Coherence (AUC): {transfer_result['coherence_auc']:.4f}")
            print(f"  Self-accuracy: {transfer_result['self_accuracy']:.4f}")

            # Store summary result
            result = {
                'family': family_name,
                'variant': variant_name,
                'model_name': model_name,
                'model_type': model_type,
                'n_topics': len(all_topics),
                'coherence_accuracy': transfer_result['coherence_accuracy'],
                'coherence_auc': transfer_result['coherence_auc'],
                'self_accuracy': transfer_result['self_accuracy'],
            }
            all_results.append(result)

            # Save per-model checkpoint (include full transfer matrix)
            checkpoint_data = {
                'result': result,
                'transfer_matrix': transfer_result['transfer_matrix'],
                'auc_matrix': transfer_result['auc_matrix'],
                'topic_names': transfer_result['topic_names'],
            }
            save_checkpoint(checkpoint_data, EXPERIMENT_NAME, model_name)

            # Free activations
            del features_by_topic
            gc.collect()

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"exp3_coherence_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # Summary by model type
    print("\n--- Coherence by Model Type ---")
    summary = df.groupby('model_type')[['coherence_accuracy', 'coherence_auc', 'self_accuracy']].mean()
    print(summary.round(4))

    # Statistical tests
    print("\n--- Statistical Tests ---")
    for metric in ['coherence_accuracy', 'coherence_auc']:
        print(f"\n  {metric}:")
        for type1, type2 in [('base', 'instruct'), ('base', 'reasoning'), ('instruct', 'reasoning')]:
            d1 = df[df['model_type'] == type1][metric].values
            d2 = df[df['model_type'] == type2][metric].values
            if len(d1) > 0 and len(d2) > 0:
                t, p = ttest_ind(d1, d2)
                print(f"    {type1} vs {type2}: t={t:.3f}, p={p:.4f}")

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Coherence by model type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.barplot(data=df, x='model_type', y='coherence_accuracy', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('Cross-Topic Coherence (Accuracy)')
    ax.set_ylabel('Mean Off-Diagonal Transfer Accuracy')
    ax.set_xlabel('Model Type')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()

    ax = axes[1]
    sns.barplot(data=df, x='model_type', y='coherence_auc', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('Cross-Topic Coherence (AUC)')
    ax.set_ylabel('Mean Off-Diagonal Transfer AUC')
    ax.set_xlabel('Model Type')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'coherence_by_type')
    plt.close()

    # Plot 2: Coherence by model family and type
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(data=df, x='family', y='coherence_auc', hue='model_type', ax=ax,
                hue_order=['base', 'instruct', 'reasoning'])
    ax.set_title('Cross-Topic Coherence by Model Family')
    ax.set_ylabel('Coherence (AUC)')
    ax.set_xlabel('Model Family')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.legend(title='Model Type')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'coherence_by_family')
    plt.close()

    # Plot 3: Self vs Transfer accuracy
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_type in ['base', 'instruct', 'reasoning']:
        df_mt = df[df['model_type'] == model_type]
        ax.scatter(df_mt['self_accuracy'], df_mt['coherence_accuracy'],
                  label=model_type, s=100, alpha=0.7)

    ax.plot([0.4, 1], [0.4, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Self-Accuracy (train & test on same topic)')
    ax.set_ylabel('Transfer Accuracy (train on topic i, test on j)')
    ax.set_title('Self vs Cross-Topic Transfer')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'self_vs_transfer')
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 3 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_experiment()
