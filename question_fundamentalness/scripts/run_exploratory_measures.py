#!/usr/bin/env python3
"""
Exploratory Measures Script

Tries multiple implementation variations for network and tree measures:

NETWORK VARIATIONS:
1. Complete graph (no threshold) - current
2. Thresholded graph (|corr| > 0.1, 0.2, 0.3)
3. Different distance transforms (1-|corr|, 1-corr^2, -log(|corr|))

TREE VARIATIONS:
1. Root selection: max neighbor MI (current) vs max degree vs max subtree
2. Depth metrics: raw depth vs normalized vs inverse

CENTRALITY VARIATIONS:
1. Betweenness: weighted vs unweighted
2. Closeness: standard vs harmonic
3. Eigenvector vs PageRank vs Katz

Usage:
    python scripts/run_exploratory_measures.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
import networkx as nx
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import load_data_with_year_filter

warnings.filterwarnings('ignore')


def compute_correlation_matrix(df, question_cols, min_samples=30):
    """Compute pairwise Spearman correlations."""
    n = len(question_cols)
    corr_matrix = pd.DataFrame(np.nan, index=question_cols, columns=question_cols)

    for i, q1 in enumerate(question_cols):
        corr_matrix.loc[q1, q1] = 1.0
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            x = df[q1].values
            y = df[q2].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= min_samples:
                corr, _ = stats.spearmanr(x[mask], y[mask])
                corr_matrix.loc[q1, q2] = corr
                corr_matrix.loc[q2, q1] = corr

    return corr_matrix


def compute_mi_matrix(df, question_cols, min_samples=30):
    """Compute pairwise mutual information."""
    n = len(question_cols)
    mi_matrix = pd.DataFrame(0.0, index=question_cols, columns=question_cols)

    for i, q1 in enumerate(question_cols):
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            x = df[q1].values
            y = df[q2].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= min_samples:
                try:
                    mi = mutual_info_score(x[mask].astype(int), y[mask].astype(int))
                    mi_matrix.loc[q1, q2] = mi
                    mi_matrix.loc[q2, q1] = mi
                except:
                    pass

    return mi_matrix


def build_network_variants(corr_matrix, question_cols):
    """Build multiple network variants with different thresholds and transforms."""
    variants = {}

    # Variant 1: Complete graph, distance = 1 - |corr|
    G_complete = nx.Graph()
    G_complete.add_nodes_from(question_cols)
    for i, q1 in enumerate(question_cols):
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            corr = corr_matrix.loc[q1, q2]
            if np.isnan(corr):
                abs_corr = 0.01
            else:
                abs_corr = max(abs(corr), 0.01)
            G_complete.add_edge(q1, q2, weight=abs_corr, distance=1-abs_corr)
    variants['complete'] = G_complete

    # Variant 2: Thresholded graphs
    for threshold in [0.05, 0.1, 0.15, 0.2]:
        G_thresh = nx.Graph()
        G_thresh.add_nodes_from(question_cols)
        for i, q1 in enumerate(question_cols):
            for j, q2 in enumerate(question_cols):
                if j <= i:
                    continue
                corr = corr_matrix.loc[q1, q2]
                if not np.isnan(corr) and abs(corr) > threshold:
                    abs_corr = abs(corr)
                    G_thresh.add_edge(q1, q2, weight=abs_corr, distance=1-abs_corr)
        variants[f'thresh_{threshold}'] = G_thresh

    # Variant 3: Different distance transforms on complete graph
    # Transform: -log(|corr|)
    G_log = nx.Graph()
    G_log.add_nodes_from(question_cols)
    for i, q1 in enumerate(question_cols):
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            corr = corr_matrix.loc[q1, q2]
            if np.isnan(corr):
                abs_corr = 0.01
            else:
                abs_corr = max(abs(corr), 0.01)
            G_log.add_edge(q1, q2, weight=abs_corr, distance=-np.log(abs_corr))
    variants['complete_log'] = G_log

    # Transform: 1 - corr^2
    G_sq = nx.Graph()
    G_sq.add_nodes_from(question_cols)
    for i, q1 in enumerate(question_cols):
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            corr = corr_matrix.loc[q1, q2]
            if np.isnan(corr):
                corr_sq = 0.0001
            else:
                corr_sq = corr ** 2
            G_sq.add_edge(q1, q2, weight=corr_sq, distance=1-corr_sq)
    variants['complete_squared'] = G_sq

    return variants


def compute_network_centralities(G, name):
    """Compute various centrality measures for a network."""
    results = {}
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    results[f'{name}_n_edges'] = n_edges
    results[f'{name}_density'] = nx.density(G) if n_nodes > 1 else 0

    # Only compute if graph has edges
    if n_edges == 0:
        return {f'{name}_{m}': pd.Series(0, index=G.nodes())
                for m in ['betweenness', 'closeness', 'eigenvector', 'pagerank', 'degree']}

    # Betweenness (weighted by distance)
    try:
        results[f'{name}_betweenness'] = pd.Series(nx.betweenness_centrality(G, weight='distance'))
    except:
        results[f'{name}_betweenness'] = pd.Series(0, index=G.nodes())

    # Betweenness (unweighted)
    try:
        results[f'{name}_betweenness_unw'] = pd.Series(nx.betweenness_centrality(G))
    except:
        results[f'{name}_betweenness_unw'] = pd.Series(0, index=G.nodes())

    # Closeness (weighted)
    try:
        results[f'{name}_closeness'] = pd.Series(nx.closeness_centrality(G, distance='distance'))
    except:
        results[f'{name}_closeness'] = pd.Series(0, index=G.nodes())

    # Harmonic centrality
    try:
        results[f'{name}_harmonic'] = pd.Series(nx.harmonic_centrality(G, distance='distance'))
    except:
        results[f'{name}_harmonic'] = pd.Series(0, index=G.nodes())

    # Eigenvector (weighted by association)
    try:
        results[f'{name}_eigenvector'] = pd.Series(nx.eigenvector_centrality(G, weight='weight', max_iter=1000))
    except:
        results[f'{name}_eigenvector'] = pd.Series(0, index=G.nodes())

    # PageRank
    try:
        results[f'{name}_pagerank'] = pd.Series(nx.pagerank(G, weight='weight'))
    except:
        results[f'{name}_pagerank'] = pd.Series(0, index=G.nodes())

    # Degree centrality
    results[f'{name}_degree'] = pd.Series(nx.degree_centrality(G))

    # Weighted degree (strength)
    strength = {n: sum(d['weight'] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    results[f'{name}_strength'] = pd.Series(strength)

    # Katz centrality
    try:
        results[f'{name}_katz'] = pd.Series(nx.katz_centrality(G, weight='weight'))
    except:
        results[f'{name}_katz'] = pd.Series(0, index=G.nodes())

    return results


def build_tree_variants(mi_matrix, question_cols):
    """Build Chow-Liu tree with different root selection methods."""
    # Build MST
    G = nx.Graph()
    for i, q1 in enumerate(question_cols):
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            mi = mi_matrix.loc[q1, q2]
            if mi > 0:
                G.add_edge(q1, q2, weight=mi)

    if G.number_of_edges() == 0:
        return None, None

    mst = nx.maximum_spanning_tree(G, weight='weight')

    # Different root selection methods
    roots = {}

    # Method 1: Max neighbor MI sum (current)
    neighbor_mi = {n: sum(mst[n][nb]['weight'] for nb in mst.neighbors(n)) for n in mst.nodes()}
    roots['max_neighbor_mi'] = max(neighbor_mi, key=neighbor_mi.get)

    # Method 2: Max degree
    degrees = dict(mst.degree())
    roots['max_degree'] = max(degrees, key=degrees.get)

    # Method 3: Center of tree (minimize max distance to any node)
    try:
        center = nx.center(mst)[0]
        roots['center'] = center
    except:
        roots['center'] = roots['max_neighbor_mi']

    return mst, roots


def compute_tree_metrics_for_root(tree, root, question_cols, prefix):
    """Compute tree metrics with a specific root."""
    results = {}

    # Depth from root
    depths = nx.single_source_shortest_path_length(tree, root)
    results[f'{prefix}_depth'] = pd.Series(depths)

    # Inverse depth (shallower = higher score)
    max_depth = max(depths.values())
    if max_depth > 0:
        results[f'{prefix}_depth_inv'] = pd.Series({n: max_depth - d for n, d in depths.items()})
        results[f'{prefix}_depth_norm'] = pd.Series({n: 1 - d/max_depth for n, d in depths.items()})
    else:
        results[f'{prefix}_depth_inv'] = pd.Series(0, index=question_cols)
        results[f'{prefix}_depth_norm'] = pd.Series(1, index=question_cols)

    # Subtree size
    directed = nx.bfs_tree(tree, root)
    subtree_sizes = {}
    for node in tree.nodes():
        if node in directed:
            subtree_sizes[node] = len(nx.descendants(directed, node))
        else:
            subtree_sizes[node] = 0
    results[f'{prefix}_subtree'] = pd.Series(subtree_sizes)

    # Hierarchy centrality
    results[f'{prefix}_hierarchy'] = results[f'{prefix}_depth_norm'].copy()

    return results


def main():
    print("="*80)
    print("EXPLORATORY MEASURES: Multiple Implementation Variations")
    print("="*80)

    # Load data
    df, question_cols, _ = load_data_with_year_filter(min_years=2)
    print(f"\nQuestions: {len(question_cols)}")
    print(f"Respondents: {len(df)}")

    # Compute base matrices
    print("\nComputing correlation matrix...")
    corr_matrix = compute_correlation_matrix(df, question_cols)

    print("Computing MI matrix...")
    mi_matrix = compute_mi_matrix(df, question_cols)

    # Build network variants
    print("\nBuilding network variants...")
    network_variants = build_network_variants(corr_matrix, question_cols)

    # Compute centralities for each network variant
    all_results = pd.DataFrame(index=question_cols)

    for name, G in network_variants.items():
        print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        centralities = compute_network_centralities(G, name)
        for col, values in centralities.items():
            if isinstance(values, pd.Series):
                all_results[col] = values

    # Build tree variants
    print("\nBuilding tree variants...")
    mst, roots = build_tree_variants(mi_matrix, question_cols)

    if mst is not None:
        print(f"  MST: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")

        # Tree betweenness (same for all roots)
        tree_betweenness = nx.betweenness_centrality(mst)
        all_results['tree_betweenness'] = pd.Series(tree_betweenness)

        # Compute metrics for each root
        for root_method, root in roots.items():
            print(f"  Root ({root_method}): {root}")
            tree_metrics = compute_tree_metrics_for_root(mst, root, question_cols, f'tree_{root_method}')
            for col, values in tree_metrics.items():
                all_results[col] = values

    # Add MI-based measures
    all_results['mi_sum'] = mi_matrix.sum(axis=1)
    all_results['mi_mean'] = mi_matrix.mean(axis=1)
    all_results['mi_max'] = mi_matrix.max(axis=1)

    # Add correlation-based measures
    all_results['corr_abs_sum'] = corr_matrix.abs().sum(axis=1) - 1  # subtract self
    all_results['corr_abs_mean'] = (corr_matrix.abs().sum(axis=1) - 1) / (len(question_cols) - 1)

    # Save all results
    all_results.to_csv('results/exploratory_all_measures.csv')
    print(f"\nSaved {len(all_results.columns)} measures to results/exploratory_all_measures.csv")

    # Print column summary
    print("\nMeasures computed:")
    for col in sorted(all_results.columns):
        print(f"  {col}")

    print("\nDone!")


if __name__ == '__main__':
    main()
