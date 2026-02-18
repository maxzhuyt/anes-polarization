#!/usr/bin/env python3
"""
Dimensionality Reduction Analysis Script

Uses PCA and Factor Analysis to identify questions that capture
fundamental dimensions of variation in the data.

Usage:
    python scripts/run_dimensionality.py [--n_components N] [--output_prefix PREFIX]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import load_data_with_year_filter, save_results

warnings.filterwarnings('ignore')


def run_dimensionality_analysis(df, question_cols, n_components=10, min_valid_frac=0.2):
    """Run dimensionality reduction analysis."""
    print(f"\n=== Dimensionality Reduction Analysis ===")
    print(f"Questions: {len(question_cols)}")

    # Filter to columns with sufficient data
    valid_fracs = df[question_cols].notna().mean()
    valid_cols = valid_fracs[valid_fracs >= min_valid_frac].index.tolist()
    print(f"Questions with >= {min_valid_frac*100:.0f}% valid data: {len(valid_cols)}")

    if len(valid_cols) < n_components:
        n_components = len(valid_cols) - 1
        print(f"Reduced n_components to {n_components}")

    # Prepare data: impute missing with median, then standardize
    X = df[valid_cols].values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print(f"Running PCA with {n_components} components...")

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    # Get loadings (correlation of original vars with components)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=valid_cols,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    # Variance explained
    var_explained = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(n_components)],
        'variance_explained': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })

    print(f"PCA: Top 3 components explain {var_explained['cumulative_variance'].iloc[2]*100:.1f}% of variance")

    # Factor Analysis (if data permits)
    fa_loadings = None
    try:
        print(f"Running Factor Analysis...")
        fa = FactorAnalysis(n_components=min(n_components, 5), random_state=42)
        fa.fit(X_scaled)
        fa_loadings = pd.DataFrame(
            fa.components_.T,
            index=valid_cols,
            columns=[f'Factor{i+1}' for i in range(fa.n_components)]
        )
    except Exception as e:
        print(f"Factor Analysis failed: {e}")

    # Compute fundamentalness scores
    scores = pd.DataFrame(index=valid_cols)

    # PC1 loading (absolute)
    scores['pc1_loading'] = loadings['PC1'].abs()

    # Total variance captured (sum of squared loadings across top components)
    scores['total_variance_pca'] = (loadings ** 2).sum(axis=1)

    # Max loading on any component
    scores['max_loading_pca'] = loadings.abs().max(axis=1)

    # Loading entropy (complexity - does it load on one or many factors?)
    abs_loadings = loadings.abs()
    loading_props = abs_loadings.div(abs_loadings.sum(axis=1), axis=0)
    loading_entropy = -(loading_props * np.log(loading_props + 1e-10)).sum(axis=1)
    scores['loading_entropy'] = loading_entropy

    # Communality from FA if available
    if fa_loadings is not None:
        scores['communality'] = (fa_loadings ** 2).sum(axis=1)

    # Composite score
    scores['composite_dimensionality'] = (
        0.4 * (scores['total_variance_pca'] / scores['total_variance_pca'].max()) +
        0.3 * (scores['max_loading_pca'] / scores['max_loading_pca'].max()) +
        0.3 * (scores['pc1_loading'] / scores['pc1_loading'].max())
    )

    scores = scores.sort_values('composite_dimensionality', ascending=False)

    print(f"Done! Top 5 by composite_dimensionality:")
    print(scores['composite_dimensionality'].head())

    # Component interpretations
    interpretations = {}
    for i in range(min(5, n_components)):
        pc_name = f'PC{i+1}'
        top_pos = loadings[pc_name].nlargest(5)
        top_neg = loadings[pc_name].nsmallest(5)
        interpretations[pc_name] = {
            'variance_explained': float(pca.explained_variance_ratio_[i]),
            'top_positive': list(zip(top_pos.index.tolist(), top_pos.values.tolist())),
            'top_negative': list(zip(top_neg.index.tolist(), top_neg.values.tolist()))
        }

    results = {
        'scores': scores,
        'pca_loadings': loadings,
        'variance_explained': var_explained,
    }

    if fa_loadings is not None:
        results['fa_loadings'] = fa_loadings

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Dimensionality Reduction Analysis')
    parser.add_argument('--n_components', type=int, default=10, help='Number of PCA components')
    parser.add_argument('--output_prefix', type=str, default='results/dimensionality', help='Output file prefix')
    parser.add_argument('--min_years', type=int, default=2, help='Minimum years a question must appear in')
    args = parser.parse_args()

    # Load data
    df, question_cols, year_avail = load_data_with_year_filter(min_years=args.min_years)

    # Run analysis
    results = run_dimensionality_analysis(df, question_cols, n_components=args.n_components)

    # Save results
    save_results(results, args.output_prefix)

    print(f"\nDimensionality Reduction analysis complete!")


if __name__ == '__main__':
    main()
