"""
Experiment 8: Affective vs Policy Polarization

Research Question:
    Do LLMs encode "affective" (identity/feeling-based) polarization differently
    from "policy" (issue-based) polarization? Do models conflate identity with policy?

Hypotheses:
    H8a: Affective topics show higher model separation than policy topics
    H8b: Instruct models conflate affective/policy more than base models
    H8c: Cross-domain transfer (affective->policy) is higher for instruct than base

Method:
    - Use exp8_affective_topics.json (curated affective + policy topics)
    - Extract activations and compute Mahalanobis distance for each
    - Compare affective vs policy topic separation
    - Test cross-domain classifier transfer

Analysis:
    - Compare mean distances: affective vs policy by model type
    - Cross-domain transfer: train on affective, test on policy (and vice versa)
    - Correlation: affective separation vs policy separation per model
    - Representational similarity: CKA or CCA between affective and policy spaces

Runtime: ~2 hours on H100
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
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import ttest_ind, pearsonr
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

EXPERIMENT_NAME = "exp8"
PCA_DIM = 15
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

# Affective vs Policy topic classification
# Affective: topics about feelings, trust, identity, warmth
# Policy: topics about specific policy positions
AFFECTIVE_KEYWORDS = [
    'conf', 'trust', 'feel', 'like', 'fair', 'warm',
    'affect', 'close', 'opin', 'therm',
]
POLICY_KEYWORDS = [
    'ab', 'nat', 'gun', 'tax', 'cap', 'gov', 'spend',
    'lib', 'col', 'spk', 'disc', 'pol', 'grn', 'wrk',
]

def classify_topic(topic_name: str) -> str:
    """Classify topic as 'affective' or 'policy' based on name."""
    name_lower = topic_name.lower()
    for kw in AFFECTIVE_KEYWORDS:
        if kw in name_lower:
            return 'affective'
    for kw in POLICY_KEYWORDS:
        if kw in name_lower:
            return 'policy'
    return 'policy'  # Default to policy

# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 8: Affective vs Policy Polarization."""

    print("="*80)
    print("EXPERIMENT 8: AFFECTIVE VS POLICY POLARIZATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics
    print("Loading affective polarization topics...")
    with open(TOPIC_LISTS_DIR / "exp8_affective_topics.json") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} topics")

    # Classify topics
    topic_types = {name: classify_topic(name) for name in topics}
    n_affective = sum(1 for t in topic_types.values() if t == 'affective')
    n_policy = sum(1 for t in topic_types.values() if t == 'policy')
    print(f"  Affective topics: {n_affective}")
    print(f"  Policy topics: {n_policy}")

    # Load GSS data
    gss_df = load_polarization_data()

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
    all_activations = {}  # model_name -> {topic_name: activations}

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

            # Store PCA-reduced features per topic (not full 4D, saves memory)
            model_features = {}  # topic -> (N, pca_dim) reduced features
            category = "public_issues"

            for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                print(f"  [{topic_idx}/{len(topics)}] {topic_name} ({topic_types[topic_name]})")

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

                # Compute PCA and distance
                pca_result = compute_pca_and_distance(
                    activations, politician_labels, pca_dim=PCA_DIM
                )

                result = {
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': topic_name,
                    'topic_type': topic_types[topic_name],
                    'mahalanobis_dist': pca_result['mahalanobis_dist'],
                    'variance_explained': pca_result['variance_explained'],
                }

                # GSS data
                gss_row = gss_df[gss_df['variable'] == topic_name]
                if len(gss_row) > 0:
                    result['gss_polarization'] = gss_row.iloc[0]['polarization']
                else:
                    result['gss_polarization'] = np.nan

                all_results.append(result)

                # Store PCA-reduced features (tiny: 550 × 15 × 4 = 33 KB)
                model_features[topic_name] = pca_result['pca_activations']

                # Free full activations
                del activations
                torch.cuda.empty_cache()
                gc.collect()

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Cross-domain transfer test using PCA-reduced features
            print(f"\n  Computing cross-domain transfer...")
            affective_topics = [t for t in topics if topic_types[t] == 'affective']
            policy_topics = [t for t in topics if topic_types[t] == 'policy']

            if affective_topics and policy_topics:
                # Concatenate per-topic PCA features across domains
                aff_features = np.concatenate([model_features[t] for t in affective_topics], axis=1)
                pol_features = np.concatenate([model_features[t] for t in policy_topics], axis=1)

                # PCA reduce the concatenated features to PCA_DIM
                pca_aff = PCA(n_components=min(PCA_DIM, aff_features.shape[1]))
                aff_pca = pca_aff.fit_transform(StandardScaler().fit_transform(aff_features))

                pca_pol = PCA(n_components=min(PCA_DIM, pol_features.shape[1]))
                pol_pca = pca_pol.fit_transform(StandardScaler().fit_transform(pol_features))

                # Train on affective, test on policy
                clf_aff = LogisticRegression(max_iter=1000, random_state=42)
                clf_aff.fit(aff_pca, politician_labels)
                aff_to_pol_acc = accuracy_score(politician_labels, clf_aff.predict(pol_pca))

                # Train on policy, test on affective
                clf_pol = LogisticRegression(max_iter=1000, random_state=42)
                clf_pol.fit(pol_pca, politician_labels)
                pol_to_aff_acc = accuracy_score(politician_labels, clf_pol.predict(aff_pca))

                # Self-accuracy
                aff_self = accuracy_score(politician_labels, clf_aff.predict(aff_pca))
                pol_self = accuracy_score(politician_labels, clf_pol.predict(pol_pca))

                print(f"    Affective self: {aff_self:.4f}, Policy self: {pol_self:.4f}")
                print(f"    Aff->Pol transfer: {aff_to_pol_acc:.4f}")
                print(f"    Pol->Aff transfer: {pol_to_aff_acc:.4f}")

                # Add to results
                for r in all_results:
                    if r['model_name'] == model_name:
                        r['aff_to_pol_transfer'] = aff_to_pol_acc
                        r['pol_to_aff_transfer'] = pol_to_aff_acc
                        r['aff_self_accuracy'] = aff_self
                        r['pol_self_accuracy'] = pol_self

            # Checkpoint
            save_checkpoint(
                [r for r in all_results if r['model_name'] == model_name],
                EXPERIMENT_NAME, model_name
            )

            del model_features
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

    # Affective vs Policy distance comparison
    print("\n--- Mahalanobis Distance: Affective vs Policy ---")
    for model_type in ['base', 'instruct', 'reasoning']:
        df_mt = df[df['model_type'] == model_type]
        aff = df_mt[df_mt['topic_type'] == 'affective']['mahalanobis_dist']
        pol = df_mt[df_mt['topic_type'] == 'policy']['mahalanobis_dist']

        if len(aff) > 0 and len(pol) > 0:
            t, p = ttest_ind(aff, pol)
            print(f"  {model_type}: Aff mean={aff.mean():.3f}, "
                  f"Pol mean={pol.mean():.3f}, t={t:.3f}, p={p:.4f}")

    # Cross-domain transfer summary
    print("\n--- Cross-Domain Transfer ---")
    transfer_cols = ['aff_to_pol_transfer', 'pol_to_aff_transfer']
    for col in transfer_cols:
        if col in df.columns:
            summary = df.groupby('model_type')[col].first()
            print(f"  {col}:")
            print(summary.round(4))

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Distance by topic type and model type
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=df, x='model_type', y='mahalanobis_dist', hue='topic_type', ax=ax,
                order=['base', 'instruct', 'reasoning'])
    ax.set_title('Partisan Separation: Affective vs Policy Topics')
    ax.set_ylabel('Mahalanobis Distance')
    ax.set_xlabel('Model Type')
    ax.legend(title='Topic Type')

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'affective_vs_policy')
    plt.close()

    # Plot 2: By family
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, family_name in enumerate(MODEL_FAMILIES.keys()):
        ax = axes[idx]
        df_fam = df[df['family'] == family_name]

        sns.barplot(data=df_fam, x='model_type', y='mahalanobis_dist',
                   hue='topic_type', ax=ax,
                   order=['base', 'instruct', 'reasoning'])
        ax.set_title(f'{family_name}')
        ax.set_ylabel('Mahalanobis Distance')
        ax.set_xlabel('Model Type')
        ax.legend(title='Topic Type', fontsize=8)

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'affective_by_family')
    plt.close()

    # Plot 3: Cross-domain transfer
    if 'aff_to_pol_transfer' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get one row per model (transfer is same for all topics of same model)
        df_transfer = df.groupby(['model_name', 'model_type', 'family']).first().reset_index()

        x = np.arange(len(df_transfer))
        width = 0.35

        ax.bar(x - width/2, df_transfer['aff_to_pol_transfer'], width,
               label='Affective -> Policy', alpha=0.8)
        ax.bar(x + width/2, df_transfer['pol_to_aff_transfer'], width,
               label='Policy -> Affective', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(df_transfer['model_name'], rotation=45, ha='right')
        ax.set_ylabel('Transfer Accuracy')
        ax.set_title('Cross-Domain Transfer: Affective <-> Policy')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Chance')
        ax.legend()

        plt.tight_layout()
        save_figure(fig, EXPERIMENT_NAME, 'cross_domain_transfer')
        plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 8 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_experiment()
