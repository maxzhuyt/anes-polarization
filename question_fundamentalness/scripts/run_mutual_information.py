#!/usr/bin/env python3
"""
Mutual Information Analysis Script

Computes pairwise mutual information between all questions and derives
fundamentalness scores.

Usage:
    python scripts/run_mutual_information.py [--n_jobs N] [--output_prefix PREFIX]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import entropy
import multiprocessing as mp
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import load_data_with_year_filter, save_results

warnings.filterwarnings('ignore')

# Global data for workers
_MI_DATA = {}


def _init_mi_worker(data_dict):
    global _MI_DATA
    _MI_DATA = data_dict


def _compute_mi_pair(args):
    """Compute MI and NMI for a single pair."""
    col1, col2, min_samples = args
    x_full = _MI_DATA[col1]
    y_full = _MI_DATA[col2]

    mask = ~(np.isnan(x_full) | np.isnan(y_full))
    if mask.sum() < min_samples:
        return col1, col2, np.nan, np.nan

    x = x_full[mask].astype(int)
    y = y_full[mask].astype(int)

    try:
        mi = mutual_info_score(x, y)
        nmi = normalized_mutual_info_score(x, y)
        return col1, col2, mi, nmi
    except Exception:
        return col1, col2, np.nan, np.nan


def run_mi_analysis(df, question_cols, n_jobs=14, min_samples=50):
    """Run mutual information analysis."""
    print(f"\n=== Mutual Information Analysis ===")
    print(f"Questions: {len(question_cols)}")

    # Pre-extract data
    df_values = {col: df[col].values.astype(float) for col in question_cols}

    # Generate pairs (symmetric)
    pairs = []
    for i, col1 in enumerate(question_cols):
        for j, col2 in enumerate(question_cols):
            if j <= i:
                continue
            pairs.append((col1, col2, min_samples))

    print(f"Computing {len(pairs)} pairs using {n_jobs} cores...")

    # Parallel computation
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_jobs, initializer=_init_mi_worker, initargs=(df_values,)) as pool:
        results = pool.map(_compute_mi_pair, pairs)

    # Build matrices
    mi_matrix = pd.DataFrame(np.nan, index=question_cols, columns=question_cols)
    nmi_matrix = pd.DataFrame(np.nan, index=question_cols, columns=question_cols)

    for col1, col2, mi, nmi in results:
        if not np.isnan(mi):
            mi_matrix.loc[col1, col2] = mi
            mi_matrix.loc[col2, col1] = mi
            nmi_matrix.loc[col1, col2] = nmi
            nmi_matrix.loc[col2, col1] = nmi

    # Fill diagonal
    for col in question_cols:
        valid = df_values[col]
        valid = valid[~np.isnan(valid)].astype(int)
        if len(valid) > 0:
            mi_matrix.loc[col, col] = mutual_info_score(valid, valid)
            nmi_matrix.loc[col, col] = 1.0

    # Compute scores
    scores = pd.DataFrame(index=question_cols)
    scores['avg_mi'] = mi_matrix.mean(axis=1)

    mi_no_diag = mi_matrix.copy()
    np.fill_diagonal(mi_no_diag.values, np.nan)
    scores['max_mi'] = mi_no_diag.max(axis=1)
    scores['avg_nmi'] = nmi_matrix.mean(axis=1)
    scores['mi_breadth'] = (nmi_matrix > 0.1).sum(axis=1) - 1

    # Entropy
    entropies = {}
    for col in question_cols:
        valid = df_values[col]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            _, counts = np.unique(valid, return_counts=True)
            entropies[col] = entropy(counts, base=2)
        else:
            entropies[col] = np.nan
    scores['entropy'] = pd.Series(entropies)

    # Composite score (handle zero max to avoid NaN)
    def safe_normalize(s):
        max_val = s.max()
        return s / max_val if max_val > 0 else 0

    scores['composite_mi'] = (
        0.5 * safe_normalize(scores['avg_mi']) +
        0.3 * safe_normalize(scores['mi_breadth']) +
        0.2 * safe_normalize(scores['entropy'])
    )

    scores = scores.sort_values('composite_mi', ascending=False)

    print(f"Done! Top 5 by composite_mi:")
    print(scores['composite_mi'].head())

    return {
        'scores': scores,
        'mi_matrix': mi_matrix,
        'nmi_matrix': nmi_matrix
    }


def main():
    parser = argparse.ArgumentParser(description='Run Mutual Information Analysis')
    parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel workers')
    parser.add_argument('--output_prefix', type=str, default='results/mi', help='Output file prefix')
    parser.add_argument('--min_years', type=int, default=2, help='Minimum years a question must appear in')
    args = parser.parse_args()

    # Load data
    df, question_cols, year_avail = load_data_with_year_filter(min_years=args.min_years)

    # Run analysis
    results = run_mi_analysis(df, question_cols, n_jobs=args.n_jobs)

    # Save results
    save_results(results, args.output_prefix)

    print(f"\nMutual Information analysis complete!")


if __name__ == '__main__':
    main()
