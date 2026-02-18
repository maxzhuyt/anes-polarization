#!/usr/bin/env python3
"""
Tree Structure Analysis Script (Chow-Liu)

Builds optimal tree-structured approximation to the joint probability
distribution using mutual information, then analyzes tree topology
to identify fundamental questions.

Usage:
    python scripts/run_tree_structure.py [--n_jobs N] [--output_prefix PREFIX]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
import multiprocessing as mp
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import load_data_with_year_filter, save_results

warnings.filterwarnings('ignore')

# Global data for workers
_TREE_DATA = {}


def _init_tree_worker(data_dict):
    global _TREE_DATA
    _TREE_DATA = data_dict


def _compute_mi(args):
    """Compute mutual information for a single pair."""
    col1, col2, min_samples = args
    x_full = _TREE_DATA[col1]
    y_full = _TREE_DATA[col2]

    mask = ~(np.isnan(x_full) | np.isnan(y_full))
    if mask.sum() < min_samples:
        return col1, col2, 0.0  # Return 0 for missing (will be low weight)

    x = x_full[mask].astype(int)
    y = y_full[mask].astype(int)

    try:
        mi = mutual_info_score(x, y)
        return col1, col2, mi
    except Exception:
        return col1, col2, 0.0


def build_chow_liu_tree(mi_matrix, question_cols):
    """
    Build Chow-Liu tree (maximum spanning tree by MI).

    The Chow-Liu algorithm finds the optimal tree-structured approximation
    to a joint probability distribution by finding the maximum spanning tree
    where edge weights are mutual information values.
    """
    # Create complete graph with MI as weights
    G = nx.Graph()
    for i, q1 in enumerate(question_cols):
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue
            mi = mi_matrix.loc[q1, q2]
            if not np.isnan(mi) and mi > 0:
                G.add_edge(q1, q2, weight=mi)

    # Find maximum spanning tree
    mst = nx.maximum_spanning_tree(G, weight='weight')

    return mst


def compute_tree_metrics(tree, question_cols, mi_matrix):
    """Compute metrics based on tree structure."""
    scores = pd.DataFrame(index=question_cols)

    # Find the root (node with highest total MI with neighbors in tree)
    neighbor_mi = {}
    for node in tree.nodes():
        total_mi = sum(tree[node][neighbor]['weight']
                       for neighbor in tree.neighbors(node))
        neighbor_mi[node] = total_mi

    root = max(neighbor_mi, key=neighbor_mi.get)

    # Compute depth from root using BFS
    depths = nx.single_source_shortest_path_length(tree, root)
    scores['tree_depth'] = pd.Series(depths)

    # Subtree size (number of descendants)
    # Convert to directed tree rooted at root
    directed = nx.bfs_tree(tree, root)

    subtree_sizes = {}
    for node in question_cols:
        if node in directed:
            descendants = nx.descendants(directed, node)
            subtree_sizes[node] = len(descendants)
        else:
            subtree_sizes[node] = 0
    scores['subtree_size'] = pd.Series(subtree_sizes)

    # Betweenness centrality in the tree
    betweenness = nx.betweenness_centrality(tree)
    scores['tree_betweenness'] = pd.Series(betweenness)

    # Sum of MI with tree neighbors
    scores['neighbor_mi_sum'] = pd.Series(neighbor_mi)

    # Degree in tree (number of children + 1 parent, except for root)
    tree_degree = dict(tree.degree())
    scores['tree_degree'] = pd.Series(tree_degree)

    # Hierarchy centrality: based on depth (lower depth = more central)
    max_depth = scores['tree_depth'].max()
    if max_depth > 0:
        scores['hierarchy_centrality'] = 1 - (scores['tree_depth'] / max_depth)
    else:
        scores['hierarchy_centrality'] = 1.0

    # Composite tree score (handle zero max to avoid NaN)
    def safe_normalize(s):
        max_val = s.max()
        return (s / max_val).fillna(0) if max_val > 0 else pd.Series(0, index=s.index)

    scores['composite_tree'] = (
        0.3 * scores['hierarchy_centrality'].fillna(0) +
        0.3 * safe_normalize(scores['subtree_size']) +
        0.2 * safe_normalize(scores['tree_betweenness']) +
        0.2 * safe_normalize(scores['neighbor_mi_sum'])
    )

    scores = scores.sort_values('composite_tree', ascending=False)

    return scores, root


def run_tree_analysis(df, question_cols, n_jobs=14, min_samples=50):
    """Run tree structure analysis."""
    print(f"\n=== Tree Structure Analysis (Chow-Liu) ===")
    print(f"Questions: {len(question_cols)}")

    # Pre-extract data
    df_values = {col: df[col].values.astype(float) for col in question_cols}

    # Generate pairs
    pairs = []
    for i, col1 in enumerate(question_cols):
        for j, col2 in enumerate(question_cols):
            if j <= i:
                continue
            pairs.append((col1, col2, min_samples))

    print(f"Computing {len(pairs)} MI values using {n_jobs} cores...")

    # Parallel computation
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_jobs, initializer=_init_tree_worker, initargs=(df_values,)) as pool:
        results = pool.map(_compute_mi, pairs)

    # Build MI matrix
    mi_matrix = pd.DataFrame(0.0, index=question_cols, columns=question_cols)
    for col1, col2, mi in results:
        mi_matrix.loc[col1, col2] = mi
        mi_matrix.loc[col2, col1] = mi

    print("Building Chow-Liu tree (maximum spanning tree by MI)...")
    tree = build_chow_liu_tree(mi_matrix, question_cols)
    print(f"Tree: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges")

    print("Computing tree-based metrics...")
    scores, root = compute_tree_metrics(tree, question_cols, mi_matrix)

    print(f"Tree root (highest neighbor MI): {root}")
    print(f"Done! Top 5 by composite_tree:")
    print(scores['composite_tree'].head())

    # Get tree edges for output
    tree_edges = []
    for u, v, d in tree.edges(data=True):
        tree_edges.append({
            'source': u,
            'target': v,
            'mi': d['weight']
        })
    tree_edges_df = pd.DataFrame(tree_edges).sort_values('mi', ascending=False)

    return {
        'scores': scores,
        'mi_matrix': mi_matrix,
        'tree_edges': tree_edges_df,
        'tree_root': {'root': root}
    }


def main():
    parser = argparse.ArgumentParser(description='Run Tree Structure Analysis')
    parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel workers')
    parser.add_argument('--output_prefix', type=str, default='results/tree', help='Output file prefix')
    parser.add_argument('--min_years', type=int, default=2, help='Minimum years a question must appear in')
    args = parser.parse_args()

    # Load data
    df, question_cols, year_avail = load_data_with_year_filter(min_years=args.min_years)

    # Run analysis
    results = run_tree_analysis(df, question_cols, n_jobs=args.n_jobs)

    # Save results
    save_results(results, args.output_prefix)

    print(f"\nTree Structure analysis complete!")


if __name__ == '__main__':
    main()
