"""
Experiment 9: Name Anonymization Test

Research Question:
    Is the partisan signal in LLM activations driven by politician name recognition
    or by political content/context?

Hypotheses:
    H9a: Named condition shows higher Mahalanobis distance than anonymous condition
    H9b: The named-anonymous gap is larger for instruct models (which "know" more about politicians)
    H9c: If anonymous distance >> chance, then political CONTENT (not just names) drives encoding

Method:
    - Two conditions per topic:
      1. "Named": real politician names (e.g., "Nancy Pelosi")
      2. "Anonymous": generic party labels (e.g., "a Democratic politician", "a Republican politician")
    - Same 550 D/R labels, same topics, same models
    - Compare Mahalanobis distances between conditions
    - If anonymous ≈ named → names are just a label, content matters
    - If anonymous << named → identity recognition dominates

Design:
    - 30 most polarized topics (same as exp5)
    - 8 models (base/instruct/reasoning)
    - Within-model paired comparison (same topic, named vs anonymous)

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
from scipy.stats import ttest_rel, ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests

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

EXPERIMENT_NAME = "exp9"
PCA_DIMS = [5, 10, 15]
MAX_LENGTH = 128

# Anonymous labels: one per real politician, preserving party balance
# We use generic numbered labels so each "person" is distinct
def make_anonymous_names(party_labels):
    """Create anonymous names preserving party identity."""
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

# Templates for anonymous condition
ANON_BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}
ANON_INSTRUCT_TEMPLATES = {
    "public_issues": "Generate a statement by {name} on {topic}.",
    "private_life":  "What would {name} say about {topic}?",
}

# =============================================================================
# Topic Loading
# =============================================================================

def load_experiment_topics() -> Dict[str, Dict]:
    """Load top-30 most polarized topics (same as exp5)."""
    topic_file = TOPIC_LISTS_DIR / "exp5_polarized_topics.json"
    with open(topic_file) as f:
        topics = json.load(f)
    # All 30 topics in one category
    return {'all_topics': topics}


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    print("="*80)
    print("EXPERIMENT 9: NAME ANONYMIZATION TEST")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # Load topics
    print("Loading topics...")
    topics_by_category = load_experiment_topics()
    total_topics = sum(len(topics) for topics in topics_by_category.values())
    print(f"Loaded {total_topics} topics")

    # Load politicians
    print("\nLoading politicians...")
    politician_csv = "/project/jevans/maxzhuyt/data/HS116_members_fullname.csv"
    df_politicians = load_politicians(politician_csv)
    politician_names = df_politicians['fullname'].tolist()
    politician_labels = (df_politicians['party_code'].values == 200).astype(int)

    n_dem = int(np.sum(politician_labels == 0))
    n_rep = int(np.sum(politician_labels == 1))
    print(f"Loaded {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Create anonymous names (same party balance)
    anon_names = make_anonymous_names(politician_labels)
    print(f"Created {len(anon_names)} anonymous names")
    print(f"  Example D: {anon_names[0]}")
    print(f"  Example R: {anon_names[politician_labels.tolist().index(1)]}")

    # Results storage
    all_results = []

    # Prompt templates
    BASE_TEMPLATES = {
        "all_topics": "{name} makes a statement on {topic}:",
    }
    INSTRUCT_TEMPLATE_KEYS = {
        "all_topics": "default",
    }

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

            # Load model ONCE
            model, tokenizer = load_model(model_path)
            if model_type == "base":
                tokenizer.chat_template = None

            for category, topics in topics_by_category.items():
                for topic_idx, (topic_name, topic_desc) in enumerate(topics.items(), 1):
                    print(f"  [{topic_idx}/{len(topics)}] {topic_name}")

                    # === CONDITION 1: NAMED ===
                    if model_type == "base":
                        template = BASE_TEMPLATES[category]
                        named_prompts = [template.format(name=name, topic=topic_desc)
                                         for name in politician_names]
                        system_msg = ""
                    else:
                        template_key = INSTRUCT_TEMPLATE_KEYS[category]
                        template = POLITICIAN_TEMPLATES[template_key]
                        named_prompts = generate_politician_prompts(
                            topic_desc, politician_names, template=template
                        )
                        system_msg = SYSTEM_MSG_POLITICIAN

                    print(f"    Named condition ({len(named_prompts)} prompts)...")
                    named_acts = extract_heads_batched(
                        model, tokenizer, named_prompts, system_msg,
                        batch_size=batch_size, max_length=MAX_LENGTH,
                    )

                    # === CONDITION 2: ANONYMOUS ===
                    if model_type == "base":
                        anon_prompts = [template.format(name=name, topic=topic_desc)
                                        for name in anon_names]
                        anon_system_msg = ""
                    else:
                        anon_prompts = generate_politician_prompts(
                            topic_desc, anon_names, template=template
                        )
                        anon_system_msg = SYSTEM_MSG_POLITICIAN

                    print(f"    Anonymous condition ({len(anon_prompts)} prompts)...")
                    anon_acts = extract_heads_batched(
                        model, tokenizer, anon_prompts, anon_system_msg,
                        batch_size=batch_size, max_length=MAX_LENGTH,
                    )

                    # Compute PCA/Mahalanobis for both conditions
                    for pca_dim in PCA_DIMS:
                        named_results = compute_pca_and_distance(
                            named_acts, politician_labels, pca_dim=pca_dim
                        )
                        anon_results = compute_pca_and_distance(
                            anon_acts, politician_labels, pca_dim=pca_dim
                        )

                        all_results.append({
                            'family': family_name,
                            'variant': variant_name,
                            'model_name': model_name,
                            'model_type': model_type,
                            'topic_name': topic_name,
                            'pca_dim': pca_dim,
                            'named_mahal': named_results['mahalanobis_dist'],
                            'named_var_pc1': named_results['variance_explained'],
                            'anon_mahal': anon_results['mahalanobis_dist'],
                            'anon_var_pc1': anon_results['variance_explained'],
                            'mahal_ratio': (named_results['mahalanobis_dist'] /
                                            max(anon_results['mahalanobis_dist'], 1e-6)),
                            'mahal_diff': (named_results['mahalanobis_dist'] -
                                           anon_results['mahalanobis_dist']),
                        })

                    # Free memory after both conditions
                    del named_acts, anon_acts
                    torch.cuda.empty_cache()
                    gc.collect()

            # Unload model
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Save per-model checkpoint
            model_results = [r for r in all_results if r['model_name'] == model_name]
            save_checkpoint(model_results, EXPERIMENT_NAME, model_name)
            print(f"Saved checkpoint for {model_name}")

    # ==========================================================================
    # Analysis
    # ==========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"exp9_anonymization_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results: {csv_path}")

    # --- Key analysis at PCA=15 ---
    df15 = df[df['pca_dim'] == 15].copy()

    print("\n--- Mahalanobis Distance: Named vs Anonymous ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df15[df15['model_type'] == mt]
        if len(sub) == 0:
            continue
        named_mean = sub['named_mahal'].mean()
        anon_mean = sub['anon_mahal'].mean()
        ratio = sub['mahal_ratio'].mean()

        # Paired t-test (same topic, same model)
        t_stat, p_val = ttest_rel(sub['named_mahal'], sub['anon_mahal'])

        # Effect size (Cohen's d for paired data)
        diff = sub['named_mahal'] - sub['anon_mahal']
        d = diff.mean() / diff.std() if diff.std() > 0 else 0

        print(f"  {mt}: Named={named_mean:.3f}, Anon={anon_mean:.3f}, "
              f"Ratio={ratio:.2f}x, t={t_stat:.3f}, p={p_val:.4e}, d={d:.3f}")

    # --- Does anonymous condition still show above-chance separation? ---
    print("\n--- Anonymous Condition: Above Chance? ---")
    print("  (Chance Mahalanobis ≈ 0 if no party signal)")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df15[df15['model_type'] == mt]
        if len(sub) == 0:
            continue
        anon_mean = sub['anon_mahal'].mean()
        anon_std = sub['anon_mahal'].std()
        # One-sample t-test vs 0
        from scipy.stats import ttest_1samp
        t_stat, p_val = ttest_1samp(sub['anon_mahal'], 0)
        print(f"  {mt}: Anon Mahalanobis = {anon_mean:.3f} +/- {anon_std:.3f}, "
              f"t={t_stat:.3f}, p={p_val:.4e}")

    # --- Named-Anonymous gap by model type ---
    print("\n--- Gap Size by Model Type ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df15[df15['model_type'] == mt]
        if len(sub) == 0:
            continue
        gap = sub['mahal_diff'].mean()
        gap_std = sub['mahal_diff'].std()
        pct_from_name = (gap / sub['named_mahal'].mean() * 100) if sub['named_mahal'].mean() > 0 else 0
        print(f"  {mt}: Gap = {gap:.3f} +/- {gap_std:.3f} ({pct_from_name:.1f}% of named signal)")

    # --- Cross-model-type comparison of gap ---
    print("\n--- Is the gap larger for instruct models? ---")
    comparisons = [('base', 'instruct'), ('base', 'reasoning'), ('instruct', 'reasoning')]
    for mt1, mt2 in comparisons:
        gap1 = df15[df15['model_type'] == mt1]['mahal_diff'].values
        gap2 = df15[df15['model_type'] == mt2]['mahal_diff'].values
        if len(gap1) > 0 and len(gap2) > 0:
            t_stat, p_val = ttest_ind(gap1, gap2)
            print(f"  {mt1} vs {mt2} gap: t={t_stat:.3f}, p={p_val:.4e}")

    # ==========================================================================
    # Plots
    # ==========================================================================

    print("\n--- Generating Plots ---")
    setup_plot_style()

    # Plot 1: Named vs Anonymous by model type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, pca_dim in enumerate(PCA_DIMS):
        ax = axes[idx]
        sub = df[df['pca_dim'] == pca_dim]

        # Melt to long format
        named = sub[['model_type', 'named_mahal']].copy()
        named.columns = ['model_type', 'distance']
        named['condition'] = 'Named'

        anon = sub[['model_type', 'anon_mahal']].copy()
        anon.columns = ['model_type', 'distance']
        anon['condition'] = 'Anonymous'

        plot_df = pd.concat([named, anon])

        sns.boxplot(data=plot_df, x='model_type', y='distance', hue='condition',
                    ax=ax, palette=['#e74c3c', '#3498db'])
        ax.set_title(f'PCA={pca_dim}')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Mahalanobis Distance')

    plt.suptitle('Exp 9: Named vs Anonymous Partisan Encoding', fontsize=14)
    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / f"exp9_named_vs_anon_{timestamp}.png")

    # Plot 2: Gap (named - anonymous) by model type and family
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df15.copy()
    sns.boxplot(data=sub, x='family', y='mahal_diff', hue='model_type', ax=ax)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Named-Anonymous Gap by Model Family and Type')
    ax.set_xlabel('Model Family')
    ax.set_ylabel('Mahalanobis Distance Difference (Named - Anon)')
    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / f"exp9_gap_by_family_{timestamp}.png")

    # Plot 3: Scatter - named vs anonymous per topic
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, mt in enumerate(['base', 'instruct', 'reasoning']):
        ax = axes[idx]
        sub = df15[df15['model_type'] == mt]
        if len(sub) == 0:
            ax.set_title(f'{mt} (no data)')
            continue
        ax.scatter(sub['anon_mahal'], sub['named_mahal'], alpha=0.5, s=20)
        lims = [0, max(sub['named_mahal'].max(), sub['anon_mahal'].max()) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Anonymous Mahalanobis')
        ax.set_ylabel('Named Mahalanobis')
        ax.set_title(f'{mt}')
        ax.legend()

    plt.suptitle('Named vs Anonymous Distance (PCA=15)', fontsize=14)
    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / f"exp9_scatter_named_anon_{timestamp}.png")

    print(f"\n{'='*80}")
    print("EXPERIMENT 9 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_experiment()
