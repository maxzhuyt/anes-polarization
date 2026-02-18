"""
Experiment 10: Residual Topic-Specific Signal

Research Question:
    After removing the global identity signal, does meaningful topic-specific
    partisan information remain? Does cross-topic transfer improve?

Background:
    Exp3 showed cross-topic transfer at chance (~0.51), while Exp7 showed
    all heads discriminate party at AUC ~0.96. This paradox suggests a global
    identity signal (name recognition) masks any topic-specific political content.

Method:
    1. For each model, extract activations for 20 topics
    2. Compute GLOBAL party centroids (mean D activation, mean R activation across ALL topics)
    3. Subtract global centroids from each topic's activations → residual activations
    4. Recompute Mahalanobis distance on residuals (topic-specific signal strength)
    5. Recompute cross-topic transfer on residuals (ideological coherence)
    6. Compare: if transfer improves, there IS shared ideology beneath identity

Hypotheses:
    H10a: Residual Mahalanobis distance is smaller than original (identity signal removed)
    H10b: Residual cross-topic transfer improves over chance (shared ideological structure)
    H10c: Instruct models show more residual transfer (richer political representations)

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
import torch
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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

EXPERIMENT_NAME = "exp10"
PCA_DIM = 15  # Use 15 dims for main analysis
MAX_LENGTH = 128
N_TOPICS = 20  # Fewer topics to fit in memory (we need to hold ALL topics' reduced features)


def load_experiment_topics() -> Dict[str, str]:
    """Load 20 topics (10 public + 10 private) for cross-topic analysis."""
    # Use exp2 topic lists (20+20 format)
    with open(TOPIC_LISTS_DIR / "exp2_public.json") as f:
        public = json.load(f)
    with open(TOPIC_LISTS_DIR / "exp2_private.json") as f:
        private = json.load(f)

    # Take first 10 from each
    topics = {}
    for i, (k, v) in enumerate(public.items()):
        if i >= 10:
            break
        topics[k] = v
    for i, (k, v) in enumerate(private.items()):
        if i >= 10:
            break
        topics[k] = v

    return topics


def compute_transfer_matrix(features_by_topic, party_labels):
    """Compute cross-topic transfer matrix from PCA-reduced features."""
    topic_names = list(features_by_topic.keys())
    n_topics = len(topic_names)

    transfer_matrix = np.zeros((n_topics, n_topics))
    auc_matrix = np.zeros((n_topics, n_topics))

    for i, train_topic in enumerate(topic_names):
        X_train = features_by_topic[train_topic]
        y_train = party_labels

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, y_train)

        for j, test_topic in enumerate(topic_names):
            X_test = features_by_topic[test_topic]
            y_test = party_labels

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            transfer_matrix[i, j] = accuracy_score(y_test, y_pred)
            try:
                auc_matrix[i, j] = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc_matrix[i, j] = 0.5

    mask = ~np.eye(n_topics, dtype=bool)
    coherence_acc = float(np.mean(transfer_matrix[mask]))
    coherence_auc = float(np.mean(auc_matrix[mask]))
    self_acc = float(np.mean(np.diag(transfer_matrix)))

    return {
        'coherence_accuracy': coherence_acc,
        'coherence_auc': coherence_auc,
        'self_accuracy': self_acc,
        'transfer_matrix': transfer_matrix,
        'auc_matrix': auc_matrix,
        'topic_names': topic_names,
    }


def run_experiment():
    print("="*80)
    print("EXPERIMENT 10: RESIDUAL TOPIC-SPECIFIC SIGNAL")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics
    print("Loading topics...")
    all_topics = load_experiment_topics()
    print(f"Loaded {len(all_topics)} topics")

    # Load politicians
    print("\nLoading politicians...")
    politician_csv = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
    df_politicians = load_politicians(politician_csv)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)
    n_pols = len(politician_names)

    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {n_pols} politicians (D={n_dem}, R={n_rep})")

    # Prompt templates
    BASE_TEMPLATE = "{name} makes a statement on {topic}:"
    INSTRUCT_TEMPLATE = POLITICIAN_TEMPLATES['default']

    all_results = []

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

            # ===== Phase 1: Extract PCA-reduced features for all topics =====
            # Store only PCA-reduced features (N × pca_dim) per topic
            features_by_topic = {}
            raw_acts_flat = []  # For computing global centroids

            for topic_idx, (topic_name, topic_desc) in enumerate(all_topics.items(), 1):
                print(f"  [{topic_idx}/{len(all_topics)}] {topic_name}")

                # Generate prompts
                if model_type == "base":
                    prompts = [BASE_TEMPLATE.format(name=name, topic=topic_desc)
                               for name in politician_names]
                    system_msg = ""
                else:
                    prompts = generate_politician_prompts(
                        topic_desc, politician_names, template=INSTRUCT_TEMPLATE
                    )
                    system_msg = SYSTEM_MSG_POLITICIAN

                print(f"    > Extracting with Batch Size {batch_size}...")
                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=batch_size, max_length=MAX_LENGTH,
                )

                # Flatten 4D -> 2D for this topic
                acts_2d = activations.reshape(activations.shape[0], -1)  # (N, L*H*D)
                del activations
                torch.cuda.empty_cache()
                gc.collect()

                # Standardize
                scaler = StandardScaler()
                acts_scaled = scaler.fit_transform(acts_2d)

                # PCA reduce
                pca = SkPCA(n_components=min(PCA_DIM, acts_scaled.shape[1]))
                features = pca.fit_transform(acts_scaled)  # (N, pca_dim)
                features_by_topic[topic_name] = features

                # Also accumulate for global centroid computation
                raw_acts_flat.append(acts_scaled)

                del acts_2d, acts_scaled
                gc.collect()

            # ===== Phase 2: Compute global party centroids =====
            print("\n  Computing global party centroids...")
            all_acts = np.concatenate(raw_acts_flat, axis=0)  # (N_topics * N_pols, D)
            all_labels = np.tile(politician_labels, len(all_topics))

            global_dem_centroid = all_acts[all_labels == 0].mean(axis=0)
            global_rep_centroid = all_acts[all_labels == 1].mean(axis=0)
            global_party_axis = global_rep_centroid - global_dem_centroid  # Direction vector
            global_party_axis_norm = global_party_axis / (np.linalg.norm(global_party_axis) + 1e-10)

            del all_acts, raw_acts_flat
            gc.collect()

            # ===== Phase 3: Compute residual features =====
            # For each topic, project out the global party axis
            print("  Computing residual features (projecting out global identity axis)...")
            residual_features_by_topic = {}

            for topic_name, features in features_by_topic.items():
                # Project onto global party axis
                # features is (N, pca_dim) but global axis is in full dim space
                # Instead, compute in PCA space: get the party axis direction in PCA space
                pass

            # Actually, we need to work in the original feature space for proper projection
            # But we only saved PCA-reduced features. Let's use a simpler approach:
            # Compute per-topic party centroids, subtract the GLOBAL mean centroid offset
            # This removes the average party difference but preserves topic-specific variation

            # Approach: for each topic, compute residual = feature - global_party_projection
            # where global_party_projection = (feature . global_axis) * global_axis
            # But we need this in PCA space. Since PCA was fit per-topic, the axes differ.
            #
            # SIMPLER APPROACH: Subtract per-party global mean from each observation
            # For each Democrat: residual = feature_i - global_dem_mean_in_topic_PCA_space
            # For each Republican: residual = feature_i - global_rep_mean_in_topic_PCA_space
            # Where global means are computed across ALL topics for that party

            # Compute global means in each topic's PCA space
            # (just use the party centroid across all topics for that specific PCA)
            global_dem_features = {}
            global_rep_features = {}

            # Accumulate party centroids across topics
            all_dem_feats = []
            all_rep_feats = []
            for topic_name, features in features_by_topic.items():
                all_dem_feats.append(features[politician_labels == 0])
                all_rep_feats.append(features[politician_labels == 1])

            # Global party centroids in PCA space (averaged across topics)
            global_dem_pca = np.mean(np.concatenate(all_dem_feats, axis=0), axis=0)
            global_rep_pca = np.mean(np.concatenate(all_rep_feats, axis=0), axis=0)
            global_party_axis_pca = global_rep_pca - global_dem_pca
            global_party_axis_pca_norm = global_party_axis_pca / (np.linalg.norm(global_party_axis_pca) + 1e-10)

            del all_dem_feats, all_rep_feats
            gc.collect()

            for topic_name, features in features_by_topic.items():
                # Project out the global party axis from each observation
                projections = features @ global_party_axis_pca_norm  # (N,)
                residual = features - np.outer(projections, global_party_axis_pca_norm)
                residual_features_by_topic[topic_name] = residual

            # ===== Phase 4: Compare original vs residual =====
            print("  Computing transfer matrices...")

            # Original transfer
            orig_result = compute_transfer_matrix(features_by_topic, politician_labels)
            print(f"    Original: self_acc={orig_result['self_accuracy']:.4f}, "
                  f"coherence={orig_result['coherence_accuracy']:.4f}")

            # Residual transfer
            resid_result = compute_transfer_matrix(residual_features_by_topic, politician_labels)
            print(f"    Residual: self_acc={resid_result['self_accuracy']:.4f}, "
                  f"coherence={resid_result['coherence_accuracy']:.4f}")

            # Compute per-topic Mahalanobis for both conditions
            for topic_name in all_topics:
                orig_feats = features_by_topic[topic_name]
                resid_feats = residual_features_by_topic[topic_name]

                # Original Mahalanobis
                orig_pca_result = compute_pca_and_distance(
                    orig_feats.reshape(n_pols, 1, 1, PCA_DIM),  # Fake 4D shape
                    politician_labels, pca_dim=PCA_DIM
                )
                # Residual Mahalanobis
                resid_pca_result = compute_pca_and_distance(
                    resid_feats.reshape(n_pols, 1, 1, PCA_DIM),
                    politician_labels, pca_dim=PCA_DIM
                )

                all_results.append({
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': topic_name,
                    'orig_mahal': orig_pca_result['mahalanobis_dist'],
                    'resid_mahal': resid_pca_result['mahalanobis_dist'],
                    'orig_self_acc': orig_result['self_accuracy'],
                    'orig_coherence_acc': orig_result['coherence_accuracy'],
                    'orig_coherence_auc': orig_result['coherence_auc'],
                    'resid_self_acc': resid_result['self_accuracy'],
                    'resid_coherence_acc': resid_result['coherence_accuracy'],
                    'resid_coherence_auc': resid_result['coherence_auc'],
                })

            # Cleanup
            del features_by_topic, residual_features_by_topic
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Save checkpoint
            model_results = [r for r in all_results if r['model_name'] == model_name]
            save_checkpoint(model_results, EXPERIMENT_NAME, model_name)

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"exp10_residual_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # --- Mahalanobis: Original vs Residual ---
    print("\n--- Mahalanobis Distance: Original vs Residual ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df[df['model_type'] == mt]
        if len(sub) == 0:
            continue
        orig_mean = sub['orig_mahal'].mean()
        resid_mean = sub['resid_mahal'].mean()
        reduction = (1 - resid_mean / orig_mean) * 100 if orig_mean > 0 else 0

        t_stat, p_val = ttest_rel(sub['orig_mahal'], sub['resid_mahal'])
        print(f"  {mt}: Orig={orig_mean:.3f}, Resid={resid_mean:.3f}, "
              f"Reduction={reduction:.1f}%, t={t_stat:.3f}, p={p_val:.4e}")

    # --- Cross-topic Coherence: Original vs Residual ---
    print("\n--- Cross-topic Coherence (Accuracy) ---")
    # Take first row per model (coherence is model-level, not topic-level)
    df_model = df.drop_duplicates(subset=['model_name'])
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df_model[df_model['model_type'] == mt]
        if len(sub) == 0:
            continue
        orig_coh = sub['orig_coherence_acc'].mean()
        resid_coh = sub['resid_coherence_acc'].mean()
        improvement = resid_coh - orig_coh
        print(f"  {mt}: Orig coherence={orig_coh:.4f}, Resid coherence={resid_coh:.4f}, "
              f"Improvement={improvement:+.4f}")

    # --- Is residual coherence above chance? ---
    print("\n--- Is residual coherence above chance (0.50)? ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df_model[df_model['model_type'] == mt]
        if len(sub) == 0:
            continue
        values = sub['resid_coherence_acc'].values
        if len(values) > 1:
            t_stat, p_val = ttest_1samp(values, 0.5)
            print(f"  {mt}: Resid coherence={values.mean():.4f}, t={t_stat:.3f}, p={p_val:.4e}")
        else:
            print(f"  {mt}: Resid coherence={values[0]:.4f} (n=1, no test)")

    # ==========================================================================
    # Plots
    # ==========================================================================

    print("\n--- Generating Plots ---")
    setup_plot_style()

    # Plot 1: Original vs Residual Mahalanobis
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({'model_type': row['model_type'], 'condition': 'Original',
                          'mahalanobis': row['orig_mahal']})
        plot_data.append({'model_type': row['model_type'], 'condition': 'Residual',
                          'mahalanobis': row['resid_mahal']})
    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(data=plot_df, x='model_type', y='mahalanobis', hue='condition',
                palette=['#e74c3c', '#3498db'])
    ax.set_title('Original vs Residual Mahalanobis Distance')
    ax.set_ylabel('Mahalanobis Distance')
    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / f"exp10_orig_vs_resid_{timestamp}.png")

    # Plot 2: Coherence comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    models = df_model['model_name'].values
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, df_model['orig_coherence_acc'], width, label='Original', color='#e74c3c')
    ax.bar(x + width/2, df_model['resid_coherence_acc'], width, label='Residual', color='#3498db')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Cross-topic Transfer Accuracy')
    ax.set_title('Coherence: Original vs Residual')
    ax.legend()

    # AUC
    ax = axes[1]
    ax.bar(x - width/2, df_model['orig_coherence_auc'], width, label='Original', color='#e74c3c')
    ax.bar(x + width/2, df_model['resid_coherence_auc'], width, label='Residual', color='#3498db')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Cross-topic Transfer AUC')
    ax.set_title('Coherence AUC: Original vs Residual')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / f"exp10_coherence_comparison_{timestamp}.png")

    print(f"\n{'='*80}")
    print("EXPERIMENT 10 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_experiment()
