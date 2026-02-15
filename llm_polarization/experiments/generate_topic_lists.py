"""
Generate pre-specified topic lists for all experiments.

This script creates and saves topic lists to be used in the overnight experiments.
All topic sampling is done with fixed seeds for reproducibility.
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared_utils import (
    load_all_topics,
    get_excluded_topics,
    sample_topics,
    load_polarization_data,
    compute_gss_statistics,
    TOPIC_LISTS_DIR,
    set_random_seeds
)
import numpy as np
import pandas as pd

set_random_seeds(42)

def save_topic_list(topics_dict: dict, filename: str):
    """Save topic list to JSON."""
    filepath = TOPIC_LISTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(topics_dict, f, indent=2)
    print(f"Saved: {filepath} ({len(topics_dict)} topics)")

def main():
    print("="*80)
    print("Generating Pre-Specified Topic Lists for Overnight Experiments")
    print("="*80)

    # Load all topics
    print("\nLoading all topics...")
    public_topics, private_topics = load_all_topics()
    print(f"Loaded {len(public_topics)} public topics, {len(private_topics)} private topics")

    # Get excluded topics (same as run_gss_pca.py)
    excluded = get_excluded_topics()
    print(f"Excluding {len(excluded)} problematic topics: {excluded}")

    # Load GSS data for polarization-based sampling
    print("\nLoading GSS polarization data...")
    gss_df = load_polarization_data()
    print(f"Loaded {len(gss_df)} GSS observations")

    # =========================================================================
    # Experiment 1: Random sample (seed=42)
    # =========================================================================
    print("\n--- Experiment 1: Encoding Strength ---")
    exp1_public, exp1_private = sample_topics(
        public_topics, private_topics,
        n_public=30, n_private=30,
        seed=42, exclude=excluded
    )
    save_topic_list(exp1_public, "exp1_public.json")
    save_topic_list(exp1_private, "exp1_private.json")

    # =========================================================================
    # Experiment 1R: Replication with different sample (seed=43)
    # =========================================================================
    print("\n--- Experiment 1R: Replication ---")
    exp1r_public, exp1r_private = sample_topics(
        public_topics, private_topics,
        n_public=30, n_private=30,
        seed=43, exclude=excluded
    )

    # Verify disjoint from Exp1
    overlap_public = set(exp1_public.keys()) & set(exp1r_public.keys())
    overlap_private = set(exp1_private.keys()) & set(exp1r_private.keys())
    if overlap_public or overlap_private:
        print(f"WARNING: Overlap detected! Public: {len(overlap_public)}, Private: {len(overlap_private)}")
    else:
        print("Verified: Exp1R topics are disjoint from Exp1")

    save_topic_list(exp1r_public, "exp1r_public.json")
    save_topic_list(exp1r_private, "exp1r_private.json")

    # =========================================================================
    # Experiment 2: Smaller sample for layer analysis (seed=42)
    # =========================================================================
    print("\n--- Experiment 2: Layer-Wise Depth ---")
    exp2_public, exp2_private = sample_topics(
        public_topics, private_topics,
        n_public=20, n_private=20,
        seed=42, exclude=excluded
    )
    save_topic_list(exp2_public, "exp2_public.json")
    save_topic_list(exp2_private, "exp2_private.json")

    # =========================================================================
    # Experiment 3: Core topics with GSS overlap (manual specification)
    # =========================================================================
    print("\n--- Experiment 3: Cross-Topic Coherence ---")
    # Core topics with high GSS overlap (n > 500, multiple waves 2010-2022)
    # Manually curated based on GSS documentation
    exp3_core_topics = [
        # Abortion cluster
        'abany', 'abdefect', 'abhlth', 'abnomore', 'abpoor', 'abrape', 'absingle',
        # Government spending
        'natspac', 'natenvir', 'natheal', 'natcity', 'natcrime', 'natdrug',
        'nateduc', 'natrace', 'natarms', 'nataid', 'natfare', 'natroad',
        # Racial policy
        'affrmact', 'wrkwayup', 'discaff',
        # Immigration (if available)
        # Gender/sexuality
        'marhomo', 'homosex',
        # Civil liberties
        'spkath', 'spkrac', 'spkcom', 'spkmslm',
        # Crime & justice
        'cappun', 'gunlaw',
        # Economic policy
        'eqwlth', 'tax',
        # Other social issues
        'grass', 'letdie1', 'suicide1', 'polviews',
    ]

    # Filter to topics that exist in our lists
    exp3_topics = {}
    for topic_name in exp3_core_topics:
        if topic_name in public_topics:
            exp3_topics[topic_name] = public_topics[topic_name]
        elif topic_name in private_topics:
            exp3_topics[topic_name] = private_topics[topic_name]

    print(f"Selected {len(exp3_topics)} core topics (target: 40)")

    # If we have fewer than 40, add more from high-sample topics
    if len(exp3_topics) < 40:
        # Compute sample sizes for remaining topics
        remaining_public = {k: v for k, v in public_topics.items()
                           if k not in exp3_topics and k not in excluded}

        # Add highest-sample topics to reach 40
        for topic_name in sorted(remaining_public.keys())[:40 - len(exp3_topics)]:
            exp3_topics[topic_name] = public_topics[topic_name]

    save_topic_list(exp3_topics, "exp3_core_topics.json")

    # =========================================================================
    # Experiment 4: All topics (no sampling)
    # =========================================================================
    print("\n--- Experiment 4: Moral Foundations ---")
    exp4_topics = {}

    # Sample 30 from each category, excluding problematic topics
    exp4_public, exp4_private = sample_topics(
        public_topics, private_topics,
        n_public=30, n_private=30,
        seed=42, exclude=excluded
    )

    exp4_topics.update(exp4_public)
    exp4_topics.update(exp4_private)

    save_topic_list(exp4_topics, "exp4_all_topics.json")

    # =========================================================================
    # Experiment 5: Most polarized public issues (need GSS data)
    # =========================================================================
    print("\n--- Experiment 5: Elite Amplification ---")

    # Compute polarization scores for each public topic
    topic_polarization = []

    for topic_name in public_topics.keys():
        if topic_name in excluded:
            continue

        stats = compute_gss_statistics(gss_df, topic_name)

        if stats is not None and stats['n_D'] >= 50 and stats['n_R'] >= 50:
            # Use polarization score as proxy for effect size
            # (Higher polarization = larger between-party difference)
            topic_polarization.append({
                'topic_name': topic_name,
                'polarization': stats['polarization'],
                'mean_diff': abs(stats['mean_D'] - stats['mean_R'])
            })

    # Sort by polarization score (higher = more polarized)
    topic_polarization = sorted(topic_polarization, key=lambda x: x['polarization'], reverse=True)

    # Take top 30
    exp5_topics = {item['topic_name']: public_topics[item['topic_name']]
                   for item in topic_polarization[:30]}

    print(f"Selected {len(exp5_topics)} most polarized topics (polarization range: "
          f"{topic_polarization[0]['polarization']:.3f} to {topic_polarization[29]['polarization']:.3f})")

    save_topic_list(exp5_topics, "exp5_polarized_topics.json")

    # =========================================================================
    # Experiment 6: Topics with moderate polarization (for overlap analysis)
    # =========================================================================
    print("\n--- Experiment 6: False Polarization (Overlap) ---")

    # Find topics with moderate polarization (not extremely polarized)
    # These are most likely to show within-party overlap
    overlap_topics = []

    for topic_name in list(public_topics.keys()) + list(private_topics.keys()):
        if topic_name in excluded:
            continue

        stats = compute_gss_statistics(gss_df, topic_name)

        if stats is not None and stats['n_D'] >= 50 and stats['n_R'] >= 50:
            # Use moderate polarization scores (0.2 < pol < 0.7)
            # Too low = no party difference, too high = extreme separation
            if 0.2 < stats['polarization'] < 0.7:
                overlap_topics.append({
                    'topic_name': topic_name,
                    'polarization': stats['polarization'],
                    'mean_diff': abs(stats['mean_D'] - stats['mean_R'])
                })

    # Sort by polarization (lower = more overlap potential)
    overlap_topics = sorted(overlap_topics, key=lambda x: x['polarization'], reverse=False)

    # Take 40 topics with lowest polarization (most overlap)
    exp6_topics = {}
    for item in overlap_topics[:40]:
        topic_name = item['topic_name']
        if topic_name in public_topics:
            exp6_topics[topic_name] = public_topics[topic_name]
        elif topic_name in private_topics:
            exp6_topics[topic_name] = private_topics[topic_name]

    print(f"Selected {len(exp6_topics)} topics with moderate polarization (overlap potential)")

    save_topic_list(exp6_topics, "exp6_overlap_topics.json")

    # =========================================================================
    # Experiment 7: Top 20 polarized topics (for attention analysis)
    # =========================================================================
    print("\n--- Experiment 7: Attention Flow ---")

    # Use top 20 from Exp5
    exp7_topics = {item['topic_name']: public_topics[item['topic_name']]
                   for item in topic_polarization[:20]}

    save_topic_list(exp7_topics, "exp7_attention_topics.json")

    # =========================================================================
    # Experiment 8: Random public issues for affective polarization
    # =========================================================================
    print("\n--- Experiment 8: Affective Polarization ---")

    exp8_public, _ = sample_topics(
        public_topics, private_topics,
        n_public=20, n_private=0,
        seed=42, exclude=excluded
    )

    save_topic_list(exp8_public, "exp8_affective_topics.json")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("Summary of Generated Topic Lists:")
    print("="*80)
    print(f"Exp1:  {len(exp1_public)} public + {len(exp1_private)} private (seed=42)")
    print(f"Exp1R: {len(exp1r_public)} public + {len(exp1r_private)} private (seed=43, disjoint)")
    print(f"Exp2:  {len(exp2_public)} public + {len(exp2_private)} private (seed=42)")
    print(f"Exp3:  {len(exp3_topics)} core topics (manual + high-sample)")
    print(f"Exp4:  {len(exp4_topics)} topics (30 public + 30 private)")
    print(f"Exp5:  {len(exp5_topics)} most polarized public topics")
    print(f"Exp6:  {len(exp6_topics)} topics with within-party variance")
    print(f"Exp7:  {len(exp7_topics)} top polarized topics (attention analysis)")
    print(f"Exp8:  {len(exp8_public)} public topics (affective)")
    print("="*80)

if __name__ == "__main__":
    main()
