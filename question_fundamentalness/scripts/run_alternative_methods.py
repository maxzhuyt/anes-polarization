#!/usr/bin/env python3
"""
Alternative Network and Information Methods Script

Implements three alternative approaches grounded in cognitive/sociological theory:

1. Partial Correlation Network (Direct Dependencies)
   - Uses graphical LASSO to find direct statistical relationships
   - Filters out spurious correlations through common causes

2. Ideological Proximity Network
   - Based on distance in PC-space (ideological dimensions)
   - Captures how beliefs are organized along liberal-conservative axis

3. Unique Information Contribution (Incremental Gain)
   - Greedy forward selection to find questions with unique predictive value
   - Identifies redundant vs. unique information sources

Usage:
    python scripts/run_alternative_methods.py [--n_jobs N] [--output_prefix PREFIX]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import networkx as nx
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import load_data_with_year_filter, save_results

warnings.filterwarnings('ignore')


def compute_partial_correlation_network(df, question_cols, alpha_range=None):
    """
    Compute partial correlation network using Graphical LASSO.

    Partial correlations capture direct relationships after controlling
    for all other variables, filtering out spurious correlations.

    Returns centrality measures based on the sparse precision matrix.
    """
    print("\n=== Partial Correlation Network (Graphical LASSO) ===", flush=True)

    # Prepare data: impute and standardize
    X = df[question_cols].values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print(f"  Fitting Graphical LASSO on {X_scaled.shape[1]} variables...", flush=True)

    # Fit Graphical LASSO with cross-validation for alpha selection
    if alpha_range is None:
        alpha_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    try:
        gl = GraphicalLassoCV(alphas=alpha_range, cv=5, max_iter=500, n_jobs=-1)
        gl.fit(X_scaled)
        precision = gl.precision_
        print(f"  Selected alpha: {gl.alpha_:.4f}", flush=True)
    except Exception as e:
        print(f"  Warning: GraphicalLassoCV failed ({e}), using default alpha=0.1", flush=True)
        from sklearn.covariance import GraphicalLasso
        gl = GraphicalLasso(alpha=0.1, max_iter=500)
        gl.fit(X_scaled)
        precision = gl.precision_

    # Convert precision matrix to partial correlations
    # partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)

    partial_corr_df = pd.DataFrame(partial_corr, index=question_cols, columns=question_cols)

    # Build network from partial correlations
    # Use absolute partial correlation as edge weight
    G = nx.Graph()
    for i, q1 in enumerate(question_cols):
        G.add_node(q1)
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            weight = abs(partial_corr[i, j])
            if weight > 0.01:  # Only include non-trivial edges
                G.add_edge(q1, q2, weight=weight)

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    # Compute centrality measures
    scores = pd.DataFrame(index=question_cols)

    # Degree (number of direct connections)
    scores['partial_degree'] = pd.Series(dict(G.degree()))

    # Weighted degree (strength of direct connections)
    scores['partial_strength'] = pd.Series(dict(G.degree(weight='weight')))

    # Betweenness in partial correlation network
    if G.number_of_edges() > 0:
        # Convert to distance: stronger partial corr = shorter distance
        G_dist = G.copy()
        for u, v, d in G_dist.edges(data=True):
            d['distance'] = 1.0 - d['weight']

        betweenness = nx.betweenness_centrality(G_dist, weight='distance')
        scores['partial_betweenness'] = pd.Series(betweenness)

        # Eigenvector centrality
        try:
            eigen = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            scores['partial_eigenvector'] = pd.Series(eigen)
        except:
            scores['partial_eigenvector'] = 0.0
    else:
        scores['partial_betweenness'] = 0.0
        scores['partial_eigenvector'] = 0.0

    # Composite score
    def safe_normalize(s):
        max_val = s.max()
        return s / max_val if max_val > 0 else pd.Series(0, index=s.index)

    scores['composite_partial'] = (
        0.3 * safe_normalize(scores['partial_strength']) +
        0.3 * safe_normalize(scores['partial_betweenness']) +
        0.2 * safe_normalize(scores['partial_degree']) +
        0.2 * safe_normalize(scores['partial_eigenvector'])
    )

    scores = scores.sort_values('composite_partial', ascending=False)

    print(f"  Top 5 by composite_partial:")
    print(scores['composite_partial'].head())

    return {
        'scores': scores,
        'partial_corr_matrix': partial_corr_df
    }


def compute_ideological_proximity_network(df, question_cols, n_components=5):
    """
    Compute network based on ideological proximity in PC-space.

    Questions close in ideological space (similar PC loadings) are connected.
    This captures how beliefs are organized along ideological dimensions.
    """
    print("\n=== Ideological Proximity Network (PC-space) ===", flush=True)

    # Prepare data
    X = df[question_cols].values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Fit PCA
    n_comp = min(n_components, len(question_cols) - 1, X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(X_scaled)

    print(f"  PCA: {n_comp} components, explained variance: {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    # Get loadings (correlation between variables and components)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=question_cols,
                                columns=[f'PC{i+1}' for i in range(n_comp)])

    # Compute pairwise distances in PC-space
    # Euclidean distance in loading space
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(loadings, metric='euclidean'))
    distance_df = pd.DataFrame(distances, index=question_cols, columns=question_cols)

    # Convert to similarity (proximity)
    max_dist = distances.max()
    proximity = 1 - (distances / max_dist)
    np.fill_diagonal(proximity, 0)  # No self-loops

    proximity_df = pd.DataFrame(proximity, index=question_cols, columns=question_cols)

    # Build network
    G = nx.Graph()
    for i, q1 in enumerate(question_cols):
        G.add_node(q1)
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            weight = proximity[i, j]
            if weight > 0.3:  # Only include reasonably close pairs
                G.add_edge(q1, q2, weight=weight)

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    # Compute scores
    scores = pd.DataFrame(index=question_cols)

    # PC1 loading (main ideological axis)
    scores['pc1_loading'] = loadings_df['PC1'].abs()

    # Distance from ideological centroid (mean loading position)
    centroid = loadings.mean(axis=0)
    scores['dist_from_centroid'] = np.linalg.norm(loadings - centroid, axis=1)

    # Ideological proximity: average proximity to all other questions
    scores['avg_ideological_proximity'] = proximity_df.mean(axis=1)

    # Closeness centrality in proximity network
    if G.number_of_edges() > 0:
        # For closeness, use distance (1 - proximity)
        G_dist = nx.Graph()
        for q in question_cols:
            G_dist.add_node(q)
        for i, q1 in enumerate(question_cols):
            for j, q2 in enumerate(question_cols):
                if j <= i:
                    continue
                dist = 1 - proximity[i, j]
                if dist < 0.7:  # Only connect close pairs
                    G_dist.add_edge(q1, q2, weight=dist)

        if G_dist.number_of_edges() > 0:
            closeness = nx.closeness_centrality(G_dist, distance='weight')
            scores['ideological_closeness'] = pd.Series(closeness)
        else:
            scores['ideological_closeness'] = 0.0
    else:
        scores['ideological_closeness'] = 0.0

    # Composite score
    def safe_normalize(s):
        max_val = s.max()
        return s / max_val if max_val > 0 else pd.Series(0, index=s.index)

    def safe_inverse_normalize(s):
        # Lower distance from centroid = more central
        max_val = s.max()
        if max_val > 0:
            return 1 - (s / max_val)
        return pd.Series(0.5, index=s.index)

    scores['composite_ideological'] = (
        0.3 * safe_normalize(scores['pc1_loading']) +
        0.3 * safe_normalize(scores['avg_ideological_proximity']) +
        0.2 * safe_inverse_normalize(scores['dist_from_centroid']) +
        0.2 * safe_normalize(scores['ideological_closeness'])
    )

    scores = scores.sort_values('composite_ideological', ascending=False)

    print(f"  Top 5 by composite_ideological:")
    print(scores['composite_ideological'].head())

    return {
        'scores': scores,
        'loadings': loadings_df,
        'distance_matrix': distance_df,
        'proximity_matrix': proximity_df
    }


def compute_unique_information(df, question_cols, n_jobs=1):
    """
    Compute unique information contribution via greedy forward selection.

    Questions selected early provide unique predictive value that can't
    be captured by other questions. Questions selected late are redundant.

    Uses fast correlation-based R² approximation for speed.
    """
    print("\n=== Unique Information Contribution (Greedy Selection) ===", flush=True)

    # Prepare data
    X = df[question_cols].values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    n_questions = len(question_cols)

    # Precompute correlation matrix for fast R² approximation
    # R² from regression of y on X ≈ correlation-based multiple R²
    corr_matrix = np.corrcoef(X_scaled.T)

    print(f"  Running greedy selection on {n_questions} questions...", flush=True)

    def compute_total_r2_fast(selected_indices, corr_matrix):
        """
        Fast computation of average R² using correlation matrix.
        For multiple regression y ~ X, R² = r_yx @ inv(R_xx) @ r_yx
        """
        if len(selected_indices) == 0:
            return 0.0

        non_selected = [i for i in range(corr_matrix.shape[0]) if i not in selected_indices]
        if len(non_selected) == 0:
            return 1.0

        total_r2 = 0
        selected_indices = list(selected_indices)

        # Get submatrix of correlations among predictors
        R_xx = corr_matrix[np.ix_(selected_indices, selected_indices)]

        # Regularize for stability
        R_xx += np.eye(len(selected_indices)) * 1e-6

        try:
            R_xx_inv = np.linalg.inv(R_xx)
        except:
            return 0.0

        for target_idx in non_selected:
            # Correlation between predictors and target
            r_yx = corr_matrix[target_idx, selected_indices]
            # Multiple R² = r_yx' @ R_xx^{-1} @ r_yx
            r2 = r_yx @ R_xx_inv @ r_yx
            total_r2 += max(0, min(1, r2))  # Clip to [0, 1]

        return total_r2 / len(non_selected)

    # Track selection order and marginal contributions
    selected = []
    remaining = list(range(n_questions))
    selection_order = {}
    marginal_contributions = {}

    current_r2 = 0.0

    # Greedy forward selection
    for step in range(min(n_questions, 30)):  # Select up to 30 questions
        best_gain = -np.inf
        best_idx = None

        for idx in remaining:
            candidate_selected = selected + [idx]
            new_r2 = compute_total_r2_fast(candidate_selected, corr_matrix)
            gain = new_r2 - current_r2

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None or best_gain < 1e-6:
            break

        # Record selection
        q_name = question_cols[best_idx]
        selection_order[q_name] = step + 1
        marginal_contributions[q_name] = best_gain

        selected.append(best_idx)
        remaining.remove(best_idx)
        current_r2 += best_gain

        if (step + 1) % 5 == 0:
            print(f"    Step {step + 1}: selected '{q_name}', marginal gain = {best_gain:.4f}, cumulative R² = {current_r2:.4f}", flush=True)

    # Assign remaining questions (not selected) high order numbers
    for idx in remaining:
        q_name = question_cols[idx]
        selection_order[q_name] = len(selected) + 1
        marginal_contributions[q_name] = 0.0

    # Build scores DataFrame
    scores = pd.DataFrame(index=question_cols)
    scores['selection_order'] = pd.Series(selection_order)
    scores['marginal_contribution'] = pd.Series(marginal_contributions)

    # Unique information score: inverse of selection order (earlier = higher)
    max_order = scores['selection_order'].max()
    scores['unique_info_score'] = (max_order - scores['selection_order'] + 1) / max_order

    # Composite: combine order and contribution
    def safe_normalize(s):
        max_val = s.max()
        return s / max_val if max_val > 0 else pd.Series(0, index=s.index)

    scores['composite_unique'] = (
        0.6 * scores['unique_info_score'] +
        0.4 * safe_normalize(scores['marginal_contribution'])
    )

    scores = scores.sort_values('composite_unique', ascending=False)

    print(f"\n  Top 10 by unique information:")
    print(scores[['selection_order', 'marginal_contribution', 'composite_unique']].head(10))

    return {
        'scores': scores,
        'selection_sequence': list(selection_order.keys())[:len(selected)]
    }


def main():
    parser = argparse.ArgumentParser(description='Run Alternative Network/Information Methods')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output_prefix', type=str, default='results/alternative', help='Output file prefix')
    parser.add_argument('--min_years', type=int, default=2, help='Minimum years a question must appear in')
    args = parser.parse_args()

    # Load data
    df, question_cols, year_avail = load_data_with_year_filter(min_years=args.min_years)

    print(f"\n{'='*60}")
    print("Alternative Network and Information Methods")
    print(f"{'='*60}")
    print(f"Questions: {len(question_cols)}")
    print(f"Respondents: {len(df)}")

    # Run all three methods
    results = {}

    # 1. Partial Correlation Network
    partial_results = compute_partial_correlation_network(df, question_cols)
    results['partial_scores'] = partial_results['scores']
    results['partial_corr_matrix'] = partial_results['partial_corr_matrix']

    # 2. Ideological Proximity Network
    ideological_results = compute_ideological_proximity_network(df, question_cols)
    results['ideological_scores'] = ideological_results['scores']
    results['ideological_loadings'] = ideological_results['loadings']
    results['ideological_proximity_matrix'] = ideological_results['proximity_matrix']

    # 3. Unique Information Contribution
    unique_results = compute_unique_information(df, question_cols)
    results['unique_scores'] = unique_results['scores']

    # Combine all scores into one DataFrame
    combined = pd.DataFrame(index=question_cols)
    combined['composite_partial'] = results['partial_scores']['composite_partial']
    combined['composite_ideological'] = results['ideological_scores']['composite_ideological']
    combined['composite_unique'] = results['unique_scores']['composite_unique']

    # Also include sub-metrics
    combined['partial_strength'] = results['partial_scores']['partial_strength']
    combined['partial_betweenness'] = results['partial_scores']['partial_betweenness']
    combined['pc1_loading'] = results['ideological_scores']['pc1_loading']
    combined['avg_ideological_proximity'] = results['ideological_scores']['avg_ideological_proximity']
    combined['dist_from_centroid'] = results['ideological_scores']['dist_from_centroid']
    combined['selection_order'] = results['unique_scores']['selection_order']
    combined['marginal_contribution'] = results['unique_scores']['marginal_contribution']

    results['combined_alternative'] = combined

    # Save results
    save_results(results, args.output_prefix)

    print(f"\n{'='*60}")
    print("Alternative Methods Complete!")
    print(f"{'='*60}")
    print(f"\nSaved to: {args.output_prefix}_*.csv")

    # Print correlation summary
    print("\nCorrelation between alternative composite scores:")
    corr = combined[['composite_partial', 'composite_ideological', 'composite_unique']].corr()
    print(corr.round(3).to_string())


if __name__ == '__main__':
    main()
