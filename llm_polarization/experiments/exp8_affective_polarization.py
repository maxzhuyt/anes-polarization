"""
Experiment 8: Affective vs Policy Polarization (Redesigned)

Research Question:
    Do LLMs encode affective polarization (inter-party hostility/warmth)
    differently from policy polarization (issue-position divergence)?

Background:
    In the human literature:
    - Policy polarization: divergence in issue positions between parties.
      Measured via survey items on specific policies (spending, abortion, guns).
      It is about *what you believe*.
    - Affective polarization: hostility/warmth toward the out-party irrespective
      of policy. Measured via feeling thermometers, social distance, trait ratings.
      Key work: Iyengar, Sood, & Lelkes (2012); Mason (2015).
      It is about *how you feel about the other side*.

Operationalization for LLMs:
    We use three distinct prompt types that isolate different aspects:

    1. **Policy prompts**: "What does [politician] think about [policy topic]?"
       Uses real GSS policy topics. Activations reflect policy-stance encoding.

    2. **Affective prompts**: "How does [politician] feel about [out-party]?"
       Novel prompts about inter-party feelings, trust, social distance.
       Activations reflect affective/identity-based encoding.

    3. **Identity prompts**: "Describe [politician] as a political figure."
       No policy or affective content. Activations reflect pure partisan
       identity encoding.

    The key comparison is whether D-R separation (Mahalanobis) is larger for
    affective prompts than policy prompts. If so, the model has internalized
    affective polarization beyond what policy disagreements explain.

    Cross-domain transfer (train on policy, test on affective) is standard
    and retained as a secondary analysis.

Runtime: ~3 hours on H100/A100
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    set_random_seeds,
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

EXPERIMENT_NAME = "exp8"
PCA_DIM = 15
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

# Prompt templates designed for each domain
# Policy: asks about specific issue positions
POLICY_BASE_TEMPLATE = "{name} makes a statement on {topic}:"
POLICY_INSTRUCT_TEMPLATE = "Generate a statement by {name} on {topic}."

# Affective: asks about feelings/attitudes toward the other party
AFFECTIVE_BASE_TEMPLATE = "{name} discusses {topic}:"
AFFECTIVE_INSTRUCT_TEMPLATE = "How would {name} respond when asked about {topic}?"

# Identity: asks about partisan self-concept (no policy content)
IDENTITY_BASE_TEMPLATE = "{name} describes {topic}:"
IDENTITY_INSTRUCT_TEMPLATE = "How would {name} talk about {topic}?"


# =============================================================================
# Prompt Generation
# =============================================================================

def generate_domain_prompts(
    topic_desc: str,
    politician_names: List[str],
    domain: str,
    model_type: str,
) -> tuple:
    """Generate prompts for a specific domain (policy/affective/identity).

    Returns:
        (prompts, system_msg)
    """
    if model_type == "base":
        templates = {
            "policy": POLICY_BASE_TEMPLATE,
            "affective": AFFECTIVE_BASE_TEMPLATE,
            "identity": IDENTITY_BASE_TEMPLATE,
        }
        template = templates[domain]
        prompts = [template.format(name=name, topic=topic_desc)
                   for name in politician_names]
        system_msg = ""
    else:
        templates = {
            "policy": POLICY_INSTRUCT_TEMPLATE,
            "affective": AFFECTIVE_INSTRUCT_TEMPLATE,
            "identity": IDENTITY_INSTRUCT_TEMPLATE,
        }
        template = templates[domain]
        prompts = [template.format(name=name, topic=topic_desc)
                   for name in politician_names]
        system_msg = SYSTEM_MSG_POLITICIAN

    return prompts, system_msg


def compute_cv_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Cross-validated linear probe accuracy."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scaler = StandardScaler()

    accuracies = []
    for train_idx, test_idx in skf.split(features, labels):
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        clf.fit(X_train, labels[train_idx])
        acc = accuracy_score(labels[test_idx], clf.predict(X_test))
        accuracies.append(acc)

    return np.mean(accuracies)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 8: Affective vs Policy Polarization (Redesigned)."""

    print("="*80)
    print("EXPERIMENT 8: AFFECTIVE VS POLICY POLARIZATION (REDESIGNED)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Operationalization:")
    print("  Policy:    GSS issue topics (what politicians believe)")
    print("  Affective: Inter-party feelings/trust/social distance")
    print("  Identity:  Partisan self-concept (no policy content)")
    print()

    set_random_seeds(42)

    # Load topics
    print("Loading redesigned topic lists...")
    with open(TOPIC_LISTS_DIR / "exp8_redesigned_topics.json") as f:
        all_topics = json.load(f)

    policy_topics = all_topics["policy"]
    affective_topics = all_topics["affective"]
    identity_topics = all_topics["identity"]

    print(f"  Policy topics:    {len(policy_topics)}")
    print(f"  Affective topics: {len(affective_topics)}")
    print(f"  Identity topics:  {len(identity_topics)}")

    # Load politicians
    print("\nLoading politicians...")
    df_politicians = load_politicians(POLITICIAN_CSV)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)
    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Results
    all_results = []
    domain_features = {}  # model_name -> domain -> list of (N, pca_dim) arrays

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

            model_domain_features = {"policy": [], "affective": [], "identity": []}

            # Process each domain
            all_domain_topics = [
                ("policy", policy_topics),
                ("affective", affective_topics),
                ("identity", identity_topics),
            ]

            for domain, topics in all_domain_topics:
                print(f"\n  === Domain: {domain.upper()} ({len(topics)} topics) ===")

                for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                    print(f"    [{topic_idx}/{len(topics)}] {topic_name}")

                    prompts, system_msg = generate_domain_prompts(
                        topic_desc, politician_names, domain, model_type
                    )

                    activations = extract_heads_batched(
                        model, tokenizer, prompts, system_msg,
                        batch_size=batch_size, max_length=MAX_LENGTH,
                    )

                    # Compute PCA + Mahalanobis
                    pca_result = compute_pca_and_distance(
                        activations, politician_labels, pca_dim=PCA_DIM
                    )

                    # CV probe accuracy on PCA features
                    probe_acc = compute_cv_probe_accuracy(
                        pca_result['pca_activations'], politician_labels
                    )

                    result = {
                        'family': family_name,
                        'variant': variant_name,
                        'model_name': model_name,
                        'model_type': model_type,
                        'domain': domain,
                        'topic_name': topic_name,
                        'mahalanobis_dist': pca_result['mahalanobis_dist'],
                        'variance_explained': pca_result['variance_explained'],
                        'probe_accuracy': probe_acc,
                    }
                    all_results.append(result)

                    model_domain_features[domain].append(pca_result['pca_activations'])

                    print(f"      Mahal={pca_result['mahalanobis_dist']:.3f}, "
                          f"Probe={probe_acc:.3f}")

                    del activations
                    torch.cuda.empty_cache()
                    gc.collect()

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Cross-domain transfer
            print(f"\n  Computing cross-domain transfer...")
            domains = ["policy", "affective", "identity"]
            transfer_results = {}

            for src_domain in domains:
                src_feats = np.concatenate(model_domain_features[src_domain], axis=1)
                pca_src = PCA(n_components=min(PCA_DIM, src_feats.shape[1]))
                src_pca = pca_src.fit_transform(StandardScaler().fit_transform(src_feats))

                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(src_pca, politician_labels)
                self_acc = accuracy_score(politician_labels, clf.predict(src_pca))
                transfer_results[f"{src_domain}_self"] = self_acc

                for tgt_domain in domains:
                    if tgt_domain == src_domain:
                        continue
                    tgt_feats = np.concatenate(model_domain_features[tgt_domain], axis=1)
                    pca_tgt = PCA(n_components=min(PCA_DIM, tgt_feats.shape[1]))
                    tgt_pca = pca_tgt.fit_transform(StandardScaler().fit_transform(tgt_feats))

                    # Retrain on src PCA space, apply to tgt PCA space
                    # (independent PCA spaces, so this tests shared linear structure)
                    clf_t = LogisticRegression(max_iter=1000, random_state=42)
                    clf_t.fit(src_pca, politician_labels)
                    transfer_acc = accuracy_score(politician_labels, clf_t.predict(tgt_pca))
                    transfer_results[f"{src_domain}_to_{tgt_domain}"] = transfer_acc

            print(f"    Transfer results:")
            for k, v in transfer_results.items():
                print(f"      {k}: {v:.4f}")

            # Attach transfer to all results for this model
            for r in all_results:
                if r['model_name'] == model_name:
                    r.update(transfer_results)

            domain_features[model_name] = model_domain_features

            # Checkpoint
            save_checkpoint(
                [r for r in all_results if r['model_name'] == model_name],
                EXPERIMENT_NAME, model_name
            )

            del model_domain_features
            gc.collect()

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"exp8_affective_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # --- Key comparison: Mahalanobis by domain ---
    print("\n--- Mahalanobis Distance by Domain and Model Type ---")
    print(f"{'Model Type':<12} {'Policy':>10} {'Affective':>10} {'Identity':>10}")
    print("-" * 44)
    for mt in ['base', 'instruct', 'reasoning']:
        vals = {}
        for domain in ['policy', 'affective', 'identity']:
            sub = df[(df['model_type'] == mt) & (df['domain'] == domain)]
            vals[domain] = sub['mahalanobis_dist'].mean() if len(sub) > 0 else float('nan')
        print(f"{mt:<12} {vals['policy']:>10.3f} {vals['affective']:>10.3f} {vals['identity']:>10.3f}")

    # --- Probe accuracy by domain ---
    print("\n--- Probe Accuracy by Domain and Model Type ---")
    print(f"{'Model Type':<12} {'Policy':>10} {'Affective':>10} {'Identity':>10}")
    print("-" * 44)
    for mt in ['base', 'instruct', 'reasoning']:
        vals = {}
        for domain in ['policy', 'affective', 'identity']:
            sub = df[(df['model_type'] == mt) & (df['domain'] == domain)]
            vals[domain] = sub['probe_accuracy'].mean() if len(sub) > 0 else float('nan')
        print(f"{mt:<12} {vals['policy']:>10.3f} {vals['affective']:>10.3f} {vals['identity']:>10.3f}")

    # --- Per-model breakdown ---
    print("\n--- Per-Model Mahalanobis (mean across topics) ---")
    pivot = df.groupby(['model_name', 'domain'])['mahalanobis_dist'].mean().unstack(fill_value=0)
    print(pivot.round(3).to_string())

    # --- Cross-domain transfer summary ---
    print("\n--- Cross-Domain Transfer Accuracy ---")
    transfer_cols = [c for c in df.columns if '_to_' in c or '_self' in c]
    if transfer_cols:
        df_transfer = df.groupby(['model_name', 'model_type'])[transfer_cols].first().reset_index()
        print(df_transfer[['model_name'] + transfer_cols].round(4).to_string(index=False))

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Mahalanobis by domain and model type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='model_type', y='mahalanobis_dist', hue='domain', ax=ax,
                order=['base', 'instruct', 'reasoning'],
                hue_order=['policy', 'affective', 'identity'],
                palette=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_title('Partisan Separation by Prompt Domain')
    ax.set_ylabel('Mahalanobis Distance (PCA-15)')
    ax.set_xlabel('Model Type')
    ax.legend(title='Domain')
    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'mahal_by_domain')
    plt.close()

    # Plot 2: Probe accuracy by domain and model type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='model_type', y='probe_accuracy', hue='domain', ax=ax,
                order=['base', 'instruct', 'reasoning'],
                hue_order=['policy', 'affective', 'identity'],
                palette=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_title('Linear Probe Accuracy by Prompt Domain')
    ax.set_ylabel('5-Fold CV Accuracy')
    ax.set_xlabel('Model Type')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='Chance')
    ax.legend(title='Domain')
    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'probe_by_domain')
    plt.close()

    # Plot 3: Per-family breakdown
    n_families = len(MODEL_FAMILIES)
    fig, axes = plt.subplots(1, n_families, figsize=(6*n_families, 5))
    if n_families == 1:
        axes = [axes]

    for idx, family_name in enumerate(MODEL_FAMILIES.keys()):
        ax = axes[idx]
        df_fam = df[df['family'] == family_name]

        sns.barplot(data=df_fam, x='model_type', y='mahalanobis_dist',
                   hue='domain', ax=ax,
                   order=['base', 'instruct', 'reasoning'],
                   hue_order=['policy', 'affective', 'identity'],
                   palette=['#3498db', '#e74c3c', '#2ecc71'],
                   ci=None)
        ax.set_title(f'{family_name}')
        ax.set_ylabel('Mahalanobis Distance')
        ax.set_xlabel('Model Type')
        ax.legend(title='Domain', fontsize=8)

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'mahal_by_family_domain')
    plt.close()

    # Plot 4: Cross-domain transfer heatmap (for each model type)
    if transfer_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        domains = ['policy', 'affective', 'identity']

        for idx, mt in enumerate(['base', 'instruct', 'reasoning']):
            ax = axes[idx]
            sub = df_transfer[df_transfer['model_type'] == mt]

            if len(sub) == 0:
                ax.set_title(f'{mt} (no data)')
                continue

            # Build transfer matrix
            transfer_mat = np.zeros((3, 3))
            for i, src in enumerate(domains):
                for j, tgt in enumerate(domains):
                    if i == j:
                        col = f"{src}_self"
                    else:
                        col = f"{src}_to_{tgt}"
                    if col in sub.columns:
                        transfer_mat[i, j] = sub[col].mean()

            sns.heatmap(transfer_mat, ax=ax, annot=True, fmt='.3f',
                       xticklabels=domains, yticklabels=domains,
                       vmin=0.4, vmax=1.0, cmap='RdYlGn')
            ax.set_title(f'{mt}: Train (row) → Test (col)')
            ax.set_xlabel('Test Domain')
            ax.set_ylabel('Train Domain')

        plt.suptitle('Cross-Domain Transfer Accuracy', fontsize=14)
        plt.tight_layout()
        save_figure(fig, EXPERIMENT_NAME, 'cross_domain_transfer')
        plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 8 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_experiment()
