"""
Experiment 9: Name Anonymization Test (Redesigned)

Research Question:
    Is the partisan signal in LLM activations driven by politician name recognition
    or by political content/context?

Key Design Choices (addressing original flaws):

    1. **No Mahalanobis for anonymous condition**: When using "anonymous Democrat/
       Republican", within-party variation collapses to near zero (one prototype
       per party), which artificially inflates Mahalanobis distance. Instead, we use:
       - Cosine distance between party centroids
       - Linear probe accuracy (5-fold CV)
       These metrics don't conflate reduced variance with increased separation.

    2. **Three conditions** (not just named vs anonymous):
       a. Named: real politician names ("Nancy Pelosi", "Ted Cruz")
       b. Fictional: fictional names with party labels ("John Smith, a Democratic
          senator from Ohio") — preserves within-party heterogeneity while
          removing real-world identity knowledge
       c. Anonymous: bare party labels ("Democratic politician #1") — collapses
          to pure partisan stereotype

    3. **Meaningful comparison**: If named ≈ fictional >> anonymous, then the model
       uses individual-level information but doesn't need real-world identity.
       If named >> fictional ≈ anonymous, then real-world name recognition dominates.
       If all three are similar, partisan label alone drives encoding.

Runtime: ~3 hours on H100/A100
"""

import sys
import json
import gc
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity

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

EXPERIMENT_NAME = "exp9"
PCA_DIM = 15
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

# Fictional names: diverse first/last name combinations with state/role variation
# to create within-party heterogeneity without real-world identity knowledge
FICTIONAL_FIRST_NAMES = [
    "James", "Maria", "Robert", "Patricia", "David", "Linda", "Michael", "Susan",
    "William", "Elizabeth", "Richard", "Jennifer", "Thomas", "Margaret", "Charles",
    "Sarah", "Daniel", "Karen", "Matthew", "Nancy", "Anthony", "Lisa", "Mark",
    "Betty", "Steven", "Dorothy", "Paul", "Sandra", "Andrew", "Ashley",
    "Joshua", "Emily", "Kenneth", "Donna", "Kevin", "Carol", "Brian", "Ruth",
    "George", "Sharon", "Timothy", "Michelle", "Ronald", "Laura", "Edward",
    "Kimberly", "Jason", "Deborah", "Jeffrey", "Jessica",
]

FICTIONAL_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore",
    "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Clark",
    "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Green", "Baker", "Adams", "Nelson", "Carter", "Mitchell",
    "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans",
    "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed",
    "Cook",
]

STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]

ROLES = ["senator", "representative", "congressperson"]


def make_fictional_names(party_labels: np.ndarray, seed: int = 42) -> List[str]:
    """Create fictional politician names with party labels and state/role context.

    Each fictional politician gets a unique name + party + state + role,
    preserving within-party variation.
    """
    rng = random.Random(seed)

    # Generate unique name combinations
    all_combos = []
    for first in FICTIONAL_FIRST_NAMES:
        for last in FICTIONAL_LAST_NAMES:
            all_combos.append(f"{first} {last}")
    rng.shuffle(all_combos)

    names = []
    for i, label in enumerate(party_labels):
        party_str = "Democratic" if label == 0 else "Republican"
        state = STATES[i % len(STATES)]
        role = ROLES[i % len(ROLES)]
        fictional_name = all_combos[i % len(all_combos)]
        names.append(f"{fictional_name}, a {party_str} {role} from {state}")

    return names


def make_anonymous_names(party_labels: np.ndarray) -> List[str]:
    """Create anonymous names with only party label and number."""
    names = []
    dem_count = 0
    rep_count = 0
    for label in party_labels:
        if label == 0:
            dem_count += 1
            names.append(f"Democratic politician #{dem_count}")
        else:
            rep_count += 1
            names.append(f"Republican politician #{rep_count}")
    return names


# =============================================================================
# Metrics (avoiding Mahalanobis for anonymous condition)
# =============================================================================

def compute_centroid_cosine_distance(
    activations: np.ndarray,
    labels: np.ndarray,
    pca_dim: int = 15,
) -> dict:
    """
    Compute cosine distance between party centroids in PCA space.

    Unlike Mahalanobis, cosine distance doesn't use the covariance matrix,
    so it's not inflated by reduced within-group variance.
    """
    # Flatten if 4D
    if activations.ndim > 2:
        n = activations.shape[0]
        activations = activations.reshape(n, -1)

    # PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    pca = PCA(n_components=min(pca_dim, X.shape[1]))
    X_pca = pca.fit_transform(X)

    # Centroids
    dem_mask = labels == 0
    rep_mask = labels == 1
    centroid_D = X_pca[dem_mask].mean(axis=0).reshape(1, -1)
    centroid_R = X_pca[rep_mask].mean(axis=0).reshape(1, -1)

    # Cosine distance = 1 - cosine_similarity
    cos_sim = cosine_similarity(centroid_D, centroid_R)[0, 0]
    cos_dist = 1.0 - cos_sim

    # Euclidean distance between centroids
    euclidean_dist = float(np.linalg.norm(centroid_D - centroid_R))

    return {
        'cosine_distance': cos_dist,
        'cosine_similarity': cos_sim,
        'euclidean_distance': euclidean_dist,
        'variance_explained': float(pca.explained_variance_ratio_[0]),
        'pca_activations': X_pca,
    }


def compute_cv_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> float:
    """5-fold CV linear probe accuracy."""
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

    return float(np.mean(accuracies))


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    print("="*80)
    print("EXPERIMENT 9: ANONYMIZATION TEST (REDESIGNED)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Conditions:")
    print("  1. Named:     Real politician names (Nancy Pelosi, Ted Cruz)")
    print("  2. Fictional: Fictional names + party + state (John Smith, a Democratic senator from Ohio)")
    print("  3. Anonymous: Bare party labels (Democratic politician #1)")
    print()
    print("Metrics: Cosine distance, Euclidean distance, Linear probe accuracy")
    print("         (NOT Mahalanobis — inflated for anonymous condition)")
    print()

    set_random_seeds(42)

    # Load topics (same as exp5: top-30 most polarized)
    print("Loading topics...")
    topic_file = TOPIC_LISTS_DIR / "exp5_polarized_topics.json"
    with open(topic_file) as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} topics")

    # Load politicians
    print("\nLoading politicians...")
    df_politicians = load_politicians(POLITICIAN_CSV)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)
    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Create alternative name sets
    fictional_names = make_fictional_names(politician_labels)
    anonymous_names = make_anonymous_names(politician_labels)
    print(f"  Named example:     {politician_names[0]}")
    print(f"  Fictional example: {fictional_names[0]}")
    print(f"  Anonymous example: {anonymous_names[0]}")

    # Results
    all_results = []

    # Prompt templates
    BASE_TEMPLATE = "{name} makes a statement on {topic}:"
    INSTRUCT_TEMPLATE = "Generate a statement by {name} on {topic}."

    CONDITIONS = [
        ("named", politician_names),
        ("fictional", fictional_names),
        ("anonymous", anonymous_names),
    ]

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

            for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                print(f"\n  [{topic_idx}/{len(topics)}] {topic_name}")

                for condition_name, name_list in CONDITIONS:
                    # Generate prompts
                    if model_type == "base":
                        prompts = [BASE_TEMPLATE.format(name=name, topic=topic_desc)
                                   for name in name_list]
                        system_msg = ""
                    else:
                        prompts = [INSTRUCT_TEMPLATE.format(name=name, topic=topic_desc)
                                   for name in name_list]
                        system_msg = SYSTEM_MSG_POLITICIAN

                    # Extract activations
                    activations = extract_heads_batched(
                        model, tokenizer, prompts, system_msg,
                        batch_size=batch_size, max_length=MAX_LENGTH,
                    )

                    # Compute metrics (cosine + euclidean + probe, NOT Mahalanobis)
                    dist_result = compute_centroid_cosine_distance(
                        activations, politician_labels, pca_dim=PCA_DIM
                    )

                    probe_acc = compute_cv_probe_accuracy(
                        dist_result['pca_activations'], politician_labels
                    )

                    result = {
                        'family': family_name,
                        'variant': variant_name,
                        'model_name': model_name,
                        'model_type': model_type,
                        'topic_name': topic_name,
                        'condition': condition_name,
                        'cosine_distance': dist_result['cosine_distance'],
                        'euclidean_distance': dist_result['euclidean_distance'],
                        'probe_accuracy': probe_acc,
                        'variance_explained': dist_result['variance_explained'],
                    }
                    all_results.append(result)

                    print(f"    {condition_name:>10}: cos_dist={dist_result['cosine_distance']:.4f}, "
                          f"euc={dist_result['euclidean_distance']:.3f}, "
                          f"probe={probe_acc:.3f}")

                    del activations
                    torch.cuda.empty_cache()
                    gc.collect()

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Checkpoint
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
    csv_path = RESULTS_DIR / f"exp9_anonymization_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # --- Cosine distance by condition ---
    print("\n--- Cosine Distance: Named vs Fictional vs Anonymous ---")
    print(f"{'Model Type':<12} {'Named':>10} {'Fictional':>10} {'Anonymous':>10}")
    print("-" * 44)
    for mt in ['base', 'instruct', 'reasoning']:
        vals = {}
        for cond in ['named', 'fictional', 'anonymous']:
            sub = df[(df['model_type'] == mt) & (df['condition'] == cond)]
            vals[cond] = sub['cosine_distance'].mean() if len(sub) > 0 else float('nan')
        print(f"{mt:<12} {vals['named']:>10.4f} {vals['fictional']:>10.4f} {vals['anonymous']:>10.4f}")

    # --- Probe accuracy by condition ---
    print("\n--- Linear Probe Accuracy: Named vs Fictional vs Anonymous ---")
    print(f"{'Model Type':<12} {'Named':>10} {'Fictional':>10} {'Anonymous':>10}")
    print("-" * 44)
    for mt in ['base', 'instruct', 'reasoning']:
        vals = {}
        for cond in ['named', 'fictional', 'anonymous']:
            sub = df[(df['model_type'] == mt) & (df['condition'] == cond)]
            vals[cond] = sub['probe_accuracy'].mean() if len(sub) > 0 else float('nan')
        print(f"{mt:<12} {vals['named']:>10.3f} {vals['fictional']:>10.3f} {vals['anonymous']:>10.3f}")

    # --- Per-model breakdown ---
    print("\n--- Per-Model Probe Accuracy (mean across topics) ---")
    pivot = df.groupby(['model_name', 'condition'])['probe_accuracy'].mean().unstack(fill_value=0)
    print(pivot.round(3).to_string())

    # --- Interpretation guide ---
    print("\n--- Interpretation ---")
    print("  If named ≈ fictional >> anonymous: individual variation matters, not name recognition")
    print("  If named >> fictional ≈ anonymous: real-world name recognition dominates")
    print("  If all three ≈ similar: partisan label alone drives encoding")

    # ==========================================================================
    # Plots
    # ==========================================================================

    setup_plot_style()

    # Plot 1: Probe accuracy by condition and model type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='model_type', y='probe_accuracy', hue='condition', ax=ax,
                order=['base', 'instruct', 'reasoning'],
                hue_order=['named', 'fictional', 'anonymous'],
                palette=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_title('Linear Probe Accuracy: Named vs Fictional vs Anonymous')
    ax.set_ylabel('5-Fold CV Probe Accuracy')
    ax.set_xlabel('Model Type')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='Chance')
    ax.legend(title='Condition')
    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'probe_by_condition')
    plt.close()

    # Plot 2: Cosine distance by condition and model type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='model_type', y='cosine_distance', hue='condition', ax=ax,
                order=['base', 'instruct', 'reasoning'],
                hue_order=['named', 'fictional', 'anonymous'],
                palette=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_title('Centroid Cosine Distance: Named vs Fictional vs Anonymous')
    ax.set_ylabel('Cosine Distance Between Party Centroids')
    ax.set_xlabel('Model Type')
    ax.legend(title='Condition')
    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'cosine_by_condition')
    plt.close()

    # Plot 3: Per-family breakdown
    n_families = len(MODEL_FAMILIES)
    fig, axes = plt.subplots(1, n_families, figsize=(6*n_families, 5))
    if n_families == 1:
        axes = [axes]

    for idx, family_name in enumerate(MODEL_FAMILIES.keys()):
        ax = axes[idx]
        df_fam = df[df['family'] == family_name]

        sns.barplot(data=df_fam, x='model_type', y='probe_accuracy',
                   hue='condition', ax=ax,
                   order=['base', 'instruct', 'reasoning'],
                   hue_order=['named', 'fictional', 'anonymous'],
                   palette=['#2ecc71', '#3498db', '#e74c3c'],
                   ci=None)
        ax.set_title(f'{family_name}')
        ax.set_ylabel('Probe Accuracy')
        ax.set_xlabel('Model Type')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax.legend(title='Condition', fontsize=8)

    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'probe_by_family_condition')
    plt.close()

    # Plot 4: Scatter named vs fictional (per topic)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, mt in enumerate(['base', 'instruct', 'reasoning']):
        ax = axes[idx]
        named_sub = df[(df['model_type'] == mt) & (df['condition'] == 'named')]
        fict_sub = df[(df['model_type'] == mt) & (df['condition'] == 'fictional')]

        if len(named_sub) == 0 or len(fict_sub) == 0:
            ax.set_title(f'{mt} (no data)')
            continue

        # Merge on topic
        merged = named_sub[['topic_name', 'model_name', 'probe_accuracy']].merge(
            fict_sub[['topic_name', 'model_name', 'probe_accuracy']],
            on=['topic_name', 'model_name'],
            suffixes=('_named', '_fictional')
        )

        ax.scatter(merged['probe_accuracy_fictional'],
                  merged['probe_accuracy_named'], alpha=0.5, s=20)
        lims = [0.4, 1.0]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Fictional Probe Accuracy')
        ax.set_ylabel('Named Probe Accuracy')
        ax.set_title(f'{mt}')
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.suptitle('Named vs Fictional: Per-Topic Probe Accuracy', fontsize=14)
    plt.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, 'scatter_named_vs_fictional')
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 9 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_experiment()
