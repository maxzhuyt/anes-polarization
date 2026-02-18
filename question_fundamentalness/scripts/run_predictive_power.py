#!/usr/bin/env python3
"""
Predictive Power Analysis Script

Computes pairwise predictive accuracy between all questions using
cross-validated logistic regression and ridge regression.

Usage:
    python scripts/run_predictive_power.py [--n_jobs N] [--output_prefix PREFIX]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
import multiprocessing as mp
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import load_data_with_year_filter, save_results

warnings.filterwarnings('ignore')

# Global data for workers
_PRED_DATA = {}


def _init_pred_worker(data_dict):
    global _PRED_DATA
    _PRED_DATA = data_dict


def _compute_prediction(args):
    """Compute predictive accuracy for a single pair."""
    predictor, target, min_samples = args
    x_full = _PRED_DATA[predictor]
    y_full = _PRED_DATA[target]

    mask = ~(np.isnan(x_full) | np.isnan(y_full))
    if mask.sum() < min_samples:
        return predictor, target, np.nan, np.nan, np.nan

    x = x_full[mask]
    y = y_full[mask]
    X = x.reshape(-1, 1)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return predictor, target, np.nan, np.nan, np.nan

    _, counts = np.unique(y, return_counts=True)
    baseline = counts.max() / len(y)

    acc, r2 = np.nan, np.nan

    # Classification accuracy
    try:
        n_splits = min(5, len(unique_classes))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=200, solver='lbfgs',
            multi_class='multinomial' if len(unique_classes) > 2 else 'auto', random_state=42)
        acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    except Exception:
        pass

    # R² (regression)
    try:
        r2 = max(0, cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='r2').mean())
    except Exception:
        pass

    acc_over_baseline = acc - baseline if not np.isnan(acc) else np.nan
    return predictor, target, acc, acc_over_baseline, r2


def run_predictive_analysis(df, question_cols, n_jobs=14, min_samples=100):
    """Run predictive power analysis."""
    print(f"\n=== Predictive Power Analysis ===", flush=True)
    print(f"Questions: {len(question_cols)}", flush=True)

    # Pre-extract data
    df_values = {col: df[col].values.astype(float) for col in question_cols}

    # Generate all pairs (asymmetric)
    pairs = [(pred, targ, min_samples)
             for pred in question_cols
             for targ in question_cols if pred != targ]

    print(f"Computing {len(pairs)} pairs using {n_jobs} cores...", flush=True)

    # Parallel computation with progress reporting
    ctx = mp.get_context('spawn')
    results = []
    with ctx.Pool(n_jobs, initializer=_init_pred_worker, initargs=(df_values,)) as pool:
        total = len(pairs)
        for i, result in enumerate(pool.imap_unordered(_compute_prediction, pairs, chunksize=50)):
            results.append(result)
            if (i + 1) % 500 == 0 or (i + 1) == total:
                pct = 100 * (i + 1) / total
                print(f"  Progress: {i + 1}/{total} ({pct:.1f}%)", flush=True)

    # Build matrices
    acc_matrix = pd.DataFrame(np.nan, index=question_cols, columns=question_cols)
    acc_over_baseline_matrix = pd.DataFrame(np.nan, index=question_cols, columns=question_cols)
    r2_matrix = pd.DataFrame(np.nan, index=question_cols, columns=question_cols)

    for predictor, target, acc, acc_over_base, r2 in results:
        if not np.isnan(acc):
            acc_matrix.loc[predictor, target] = acc
            acc_over_baseline_matrix.loc[predictor, target] = acc_over_base
        if not np.isnan(r2):
            r2_matrix.loc[predictor, target] = r2

    # Compute scores
    scores = pd.DataFrame(index=question_cols)
    scores['avg_predictive_acc'] = acc_matrix.mean(axis=1)
    scores['avg_acc_over_baseline'] = acc_over_baseline_matrix.mean(axis=1)
    scores['avg_predictive_r2'] = r2_matrix.mean(axis=1)
    scores['avg_predicted_acc'] = acc_matrix.mean(axis=0)
    scores['avg_predicted_r2'] = r2_matrix.mean(axis=0)
    scores['predictive_breadth'] = (acc_over_baseline_matrix > 0.05).sum(axis=1)
    scores['predictive_asymmetry'] = scores['avg_predictive_r2'] - scores['avg_predicted_r2']

    # Composite score (handle zero ranges to avoid NaN)
    def safe_normalize(s, clip_neg=False):
        max_val = s.max()
        if max_val > 0:
            result = s / max_val
            return result.clip(0, 1) if clip_neg else result
        return pd.Series(0, index=s.index)

    def safe_range_normalize(s):
        range_val = s.max() - s.min()
        if range_val > 0:
            return (s - s.min()) / range_val
        return pd.Series(0.5, index=s.index)

    scores['composite_predictive'] = (
        0.4 * safe_normalize(scores['avg_acc_over_baseline'], clip_neg=True) +
        0.3 * safe_normalize(scores['avg_predictive_r2'], clip_neg=True) +
        0.2 * safe_normalize(scores['predictive_breadth']) +
        0.1 * safe_range_normalize(scores['predictive_asymmetry'])
    )

    scores = scores.sort_values('composite_predictive', ascending=False)

    print(f"Done! Top 5 by composite_predictive:")
    print(scores['composite_predictive'].head())

    return {
        'scores': scores,
        'acc_matrix': acc_matrix,
        'r2_matrix': r2_matrix
    }


def main():
    parser = argparse.ArgumentParser(description='Run Predictive Power Analysis')
    parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel workers')
    parser.add_argument('--output_prefix', type=str, default='results/predictive', help='Output file prefix')
    parser.add_argument('--min_years', type=int, default=2, help='Minimum years a question must appear in')
    args = parser.parse_args()

    # Load data
    df, question_cols, year_avail = load_data_with_year_filter(min_years=args.min_years)

    # Run analysis
    results = run_predictive_analysis(df, question_cols, n_jobs=args.n_jobs)

    # Save results
    save_results(results, args.output_prefix)

    print(f"\nPredictive Power analysis complete!")


if __name__ == '__main__':
    main()
