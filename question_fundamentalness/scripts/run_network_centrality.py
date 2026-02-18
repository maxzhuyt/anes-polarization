#!/usr/bin/env python3
"""
Network Centrality Analysis Script

Builds a complete weighted belief network from question associations and computes
centrality measures based on geodesic (shortest-path) distances.

Key approach:
1. Compute pairwise Spearman correlations (association matrix)
2. Convert associations to distances: distance = 1 - |correlation|
   (stronger associations = shorter distances)
3. Build complete weighted network
4. Compute all-pairs shortest paths (geodesics)
5. Compute shortest-path betweenness centrality

Usage:
    python scripts/run_network_centrality.py [--n_jobs N] [--output_prefix PREFIX]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import multiprocessing as mp
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import load_data_with_year_filter, save_results

warnings.filterwarnings('ignore')

# Global data for workers
_NET_DATA = {}


def _init_net_worker(data_dict):
    global _NET_DATA
    _NET_DATA = data_dict


def _compute_correlation(args):
    """Compute Spearman correlation for a single pair."""
    col1, col2, min_samples = args
    x_full = _NET_DATA[col1]
    y_full = _NET_DATA[col2]

    mask = ~(np.isnan(x_full) | np.isnan(y_full))
    if mask.sum() < min_samples:
        return col1, col2, np.nan

    x = x_full[mask]
    y = y_full[mask]

    try:
        corr, _ = stats.spearmanr(x, y)
        return col1, col2, corr
    except Exception:
        return col1, col2, np.nan


def association_to_distance(association: float) -> float:
    """
    Convert association strength to distance.

    Stronger associations (|corr| closer to 1) -> shorter distances.
    Uses: distance = 1 - |association|

    This ensures:
    - |corr| = 1.0 -> distance = 0 (perfectly associated = no distance)
    - |corr| = 0.0 -> distance = 1 (no association = maximum distance)
    """
    if np.isnan(association):
        return np.inf  # Missing = infinite distance
    return 1.0 - abs(association)


def run_network_analysis(df, question_cols, n_jobs=14, min_samples=50):
    """Run network centrality analysis with geodesic betweenness."""
    print(f"\n=== Network Centrality Analysis ===")
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

    print(f"Computing {len(pairs)} correlations using {n_jobs} cores...")

    # Parallel computation of correlations
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_jobs, initializer=_init_net_worker, initargs=(df_values,)) as pool:
        results = pool.map(_compute_correlation, pairs)

    # Build correlation matrix
    corr_matrix = pd.DataFrame(np.nan, index=question_cols, columns=question_cols)
    for col1, col2, corr in results:
        if not np.isnan(corr):
            corr_matrix.loc[col1, col2] = corr
            corr_matrix.loc[col2, col1] = corr

    # Fill diagonal
    for col in question_cols:
        corr_matrix.loc[col, col] = 1.0

    print(f"Correlation matrix computed. Valid pairs: {sum(1 for _, _, c in results if not np.isnan(c))}")

    # Build COMPLETE weighted network
    # Edge weight = |correlation| (association strength)
    # Edge 'distance' = 1 - |correlation| (for shortest path computation)
    print("Building complete weighted network...")

    G = nx.Graph()
    G.add_nodes_from(question_cols)

    edge_count = 0
    for i, q1 in enumerate(question_cols):
        for j, q2 in enumerate(question_cols):
            if j <= i:
                continue

            corr = corr_matrix.loc[q1, q2]
            if np.isnan(corr):
                # For missing correlations, use a small positive association
                # to keep the network connected
                abs_corr = 0.01
                distance = 0.99
            else:
                abs_corr = abs(corr)
                distance = association_to_distance(corr)

            # Add edge with both weight (association) and distance
            G.add_edge(q1, q2,
                       weight=abs_corr,          # Association strength
                       distance=distance,         # For shortest paths
                       raw_correlation=corr if not np.isnan(corr) else 0.0)
            edge_count += 1

    print(f"Network: {G.number_of_nodes()} nodes, {edge_count} edges")

    # Compute all-pairs shortest paths using distance
    print("Computing all-pairs shortest paths (geodesics)...")
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='distance'))

    # Create geodesic distance matrix
    geodesic_matrix = pd.DataFrame(np.inf, index=question_cols, columns=question_cols)
    for source in question_cols:
        for target, dist in shortest_paths[source].items():
            geodesic_matrix.loc[source, target] = dist

    print("Computing centrality measures...")
    scores = pd.DataFrame(index=question_cols)

    # 1. Shortest-path betweenness centrality (using distance)
    # This measures how often a node lies on shortest paths between other nodes
    betweenness = nx.betweenness_centrality(G, weight='distance', normalized=True)
    scores['betweenness_centrality'] = pd.Series(betweenness)

    # 2. Closeness centrality (inverse of average geodesic distance)
    closeness = nx.closeness_centrality(G, distance='distance')
    scores['closeness_centrality'] = pd.Series(closeness)

    # 3. Degree centrality (fraction of nodes connected to)
    degree_cent = nx.degree_centrality(G)
    scores['degree_centrality'] = pd.Series(degree_cent)

    # 4. Strength (weighted degree using association weights)
    strength = {}
    for node in G.nodes():
        strength[node] = sum(d['weight'] for _, _, d in G.edges(node, data=True))
    scores['strength'] = pd.Series(strength)

    # 5. Eigenvector centrality (using association weights)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        scores['eigenvector_centrality'] = pd.Series(eigenvector)
    except:
        scores['eigenvector_centrality'] = np.nan

    # 6. PageRank (using association weights)
    try:
        pagerank = nx.pagerank(G, weight='weight')
        scores['pagerank'] = pd.Series(pagerank)
    except:
        scores['pagerank'] = np.nan

    # 7. Average geodesic distance (mean shortest path to all other nodes)
    avg_geodesic = geodesic_matrix.replace(np.inf, np.nan).mean(axis=1)
    scores['avg_geodesic_distance'] = avg_geodesic

    # 8. Harmonic centrality (works better for disconnected components)
    harmonic = nx.harmonic_centrality(G, distance='distance')
    scores['harmonic_centrality'] = pd.Series(harmonic)

    # Composite centrality score
    # Normalize each measure to [0, 1] and combine
    norm_cols = []
    for col in ['betweenness_centrality', 'closeness_centrality', 'strength',
                'eigenvector_centrality', 'pagerank', 'harmonic_centrality']:
        if scores[col].notna().any():
            min_val = scores[col].min()
            max_val = scores[col].max()
            if max_val > min_val:
                scores[f'{col}_norm'] = (scores[col] - min_val) / (max_val - min_val)
                norm_cols.append(f'{col}_norm')

    if norm_cols:
        # Weight betweenness and closeness more heavily (geodesic-based)
        weights = {
            'betweenness_centrality_norm': 0.25,
            'closeness_centrality_norm': 0.25,
            'strength_norm': 0.15,
            'eigenvector_centrality_norm': 0.15,
            'pagerank_norm': 0.10,
            'harmonic_centrality_norm': 0.10
        }
        composite = sum(scores[col] * weights.get(col, 0.1) for col in norm_cols if col in weights)
        scores['composite_centrality'] = composite / sum(weights.get(col, 0.1) for col in norm_cols if col in weights)
    else:
        scores['composite_centrality'] = np.nan

    scores = scores.sort_values('composite_centrality', ascending=False)

    print(f"Done! Top 5 by composite_centrality:")
    print(scores['composite_centrality'].head())

    # Network statistics
    network_stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
        'avg_clustering': nx.average_clustering(G, weight='weight'),
        'avg_shortest_path': geodesic_matrix.replace(np.inf, np.nan).mean().mean(),
    }

    if nx.is_connected(G):
        network_stats['diameter'] = nx.diameter(G, weight='distance')

    return {
        'scores': scores,
        'correlation_matrix': corr_matrix,
        'geodesic_matrix': geodesic_matrix,
        'network_stats': network_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Run Network Centrality Analysis')
    parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel workers')
    parser.add_argument('--output_prefix', type=str, default='results/network', help='Output file prefix')
    parser.add_argument('--min_years', type=int, default=2, help='Minimum years a question must appear in')
    args = parser.parse_args()

    # Load data
    df, question_cols, year_avail = load_data_with_year_filter(min_years=args.min_years)

    # Run analysis
    results = run_network_analysis(df, question_cols, n_jobs=args.n_jobs)

    # Save results
    save_results(results, args.output_prefix)

    # Print network stats
    print("\nNetwork Statistics:")
    for k, v in results['network_stats'].items():
        print(f"  {k}: {v}")

    print(f"\nNetwork Centrality analysis complete!")


if __name__ == '__main__':
    main()
