#!/usr/bin/env python3
"""
Run All Analyses Script

Runs all five analysis methods and combines results into a unified
question hierarchy ranking.

Usage:
    python scripts/run_all_analyses.py [--n_jobs N] [--output_dir DIR]
"""

import argparse
import warnings
import os
import sys
import time
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import load_data_with_year_filter
from scripts.run_mutual_information import run_mi_analysis
from scripts.run_predictive_power import run_predictive_analysis
from scripts.run_network_centrality import run_network_analysis
from scripts.run_dimensionality import run_dimensionality_analysis
from scripts.run_tree_structure import run_tree_analysis

warnings.filterwarnings('ignore')


def combine_results(results_dict, question_cols):
    """Combine scores from all methods into unified ranking."""
    print("\n=== Combining Results ===")

    # Collect composite scores
    composite_scores = pd.DataFrame(index=question_cols)

    for method, results in results_dict.items():
        if 'scores' in results:
            scores = results['scores']
            # Find the composite column
            composite_col = [c for c in scores.columns if 'composite' in c.lower()]
            if composite_col:
                composite_scores[method] = scores[composite_col[0]].reindex(question_cols)

    # Fill NaN with 0
    composite_scores = composite_scores.fillna(0)

    # Compute ranks
    ranks = composite_scores.rank(ascending=False)
    composite_scores['avg_rank'] = ranks.mean(axis=1)
    composite_scores['rank_std'] = ranks.std(axis=1)

    # Ensemble score (mean of normalized scores)
    method_cols = [c for c in composite_scores.columns if c not in ['avg_rank', 'rank_std']]
    normalized = composite_scores[method_cols].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )
    composite_scores['ensemble_score'] = normalized.mean(axis=1)

    # Tier assignment
    composite_scores['tier'] = pd.cut(
        composite_scores['ensemble_score'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Tier 4 (Specific)', 'Tier 3', 'Tier 2', 'Tier 1 (Fundamental)']
    )

    composite_scores = composite_scores.sort_values('ensemble_score', ascending=False)

    return composite_scores


def main():
    parser = argparse.ArgumentParser(description='Run All Analyses')
    parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--min_years', type=int, default=2, help='Minimum years a question must appear in')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data once
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    df, question_cols, year_avail = load_data_with_year_filter(min_years=args.min_years)

    # Save question list
    pd.DataFrame({
        'question': question_cols,
        'years': [year_avail[q] for q in question_cols]
    }).to_csv(f"{args.output_dir}/questions.csv", index=False)

    all_results = {}
    timings = {}

    # 1. Mutual Information
    print("\n" + "=" * 60)
    start = time.time()
    all_results['MI'] = run_mi_analysis(df, question_cols, n_jobs=args.n_jobs)
    timings['MI'] = time.time() - start
    for name, data in all_results['MI'].items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{args.output_dir}/mi_{name}.csv")

    # 2. Predictive Power
    print("\n" + "=" * 60)
    start = time.time()
    all_results['Predictive'] = run_predictive_analysis(df, question_cols, n_jobs=args.n_jobs)
    timings['Predictive'] = time.time() - start
    for name, data in all_results['Predictive'].items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{args.output_dir}/predictive_{name}.csv")

    # 3. Network Centrality
    print("\n" + "=" * 60)
    start = time.time()
    all_results['Network'] = run_network_analysis(df, question_cols, n_jobs=args.n_jobs)
    timings['Network'] = time.time() - start
    for name, data in all_results['Network'].items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{args.output_dir}/network_{name}.csv")

    # 4. Dimensionality Reduction
    print("\n" + "=" * 60)
    start = time.time()
    all_results['Dimensionality'] = run_dimensionality_analysis(df, question_cols)
    timings['Dimensionality'] = time.time() - start
    for name, data in all_results['Dimensionality'].items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{args.output_dir}/dimensionality_{name}.csv")

    # 5. Tree Structure
    print("\n" + "=" * 60)
    start = time.time()
    all_results['Tree'] = run_tree_analysis(df, question_cols, n_jobs=args.n_jobs)
    timings['Tree'] = time.time() - start
    for name, data in all_results['Tree'].items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{args.output_dir}/tree_{name}.csv")

    # Combine results
    print("\n" + "=" * 60)
    combined = combine_results(all_results, question_cols)
    combined.to_csv(f"{args.output_dir}/combined_hierarchy.csv")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print("\nTimings:")
    for method, t in timings.items():
        print(f"  {method}: {t:.1f}s")
    print(f"  Total: {sum(timings.values()):.1f}s")

    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - combined_hierarchy.csv (unified ranking)")
    print(f"  - mi_*.csv (mutual information)")
    print(f"  - predictive_*.csv (predictive power)")
    print(f"  - network_*.csv (network centrality)")
    print(f"  - dimensionality_*.csv (PCA/FA)")
    print(f"  - tree_*.csv (Chow-Liu tree)")

    print("\nTop 10 Most Fundamental Questions:")
    print(combined[['ensemble_score', 'avg_rank', 'tier']].head(10).to_string())


if __name__ == '__main__':
    main()
