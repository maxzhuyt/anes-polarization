"""
Network Centrality approach to measuring question fundamentalness.

Theory:
- Build a network where questions are nodes and edges represent relationships
- Edge weights can be correlation, mutual information, or predictive power
- Questions with high "centrality" in this network are fundamental

Centrality measures:
1. Degree centrality: How many questions is this connected to?
2. Weighted degree (strength): Sum of edge weights
3. Betweenness centrality: How often is this question on shortest paths?
4. Eigenvector centrality: Is this question connected to other important questions?
5. PageRank: Importance based on the importance of neighbors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import networkx as nx

from .utils import get_valid_pairs, compute_pairwise_matrix, DEFAULT_N_JOBS


class NetworkCentralityAnalyzer:
    """
    Analyze question fundamentalness using network centrality measures.
    """

    def __init__(self, df: pd.DataFrame, question_cols: List[str]):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed GSS data
        question_cols : List[str]
            List of question column names
        """
        self.df = df
        self.question_cols = question_cols
        self.correlation_matrix = None
        self.graph = None
        self.scores = None

    def _compute_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman correlation (robust to non-linearity)."""
        corr, _ = stats.spearmanr(x, y)
        return corr

    def _compute_abs_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute absolute Spearman correlation."""
        corr, _ = stats.spearmanr(x, y)
        return abs(corr)

    def fit(
        self,
        edge_type: str = 'correlation',
        edge_threshold: float = 0.1,
        verbose: bool = True,
        n_jobs: int = DEFAULT_N_JOBS
    ) -> 'NetworkCentralityAnalyzer':
        """
        Build the question network and compute centrality measures.

        Parameters
        ----------
        edge_type : str
            'correlation' for Spearman correlation (default)
        edge_threshold : float
            Minimum absolute edge weight to include (sparsify the network)
        verbose : bool
            Print progress
        n_jobs : int
            Number of parallel jobs (default: 14)
        """
        if verbose:
            print(f"Computing correlation matrix using {n_jobs} cores...")

        # Compute correlation matrix
        self.correlation_matrix = compute_pairwise_matrix(
            self.df, self.question_cols, self._compute_correlation,
            symmetric=True, verbose=verbose, n_jobs=n_jobs
        )

        # Build networkx graph
        if verbose:
            print("Building network graph...")

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.question_cols)

        for i, q1 in enumerate(self.question_cols):
            for j, q2 in enumerate(self.question_cols):
                if j <= i:
                    continue

                weight = self.correlation_matrix.loc[q1, q2]
                if np.isnan(weight):
                    continue

                abs_weight = abs(weight)
                if abs_weight >= edge_threshold:
                    self.graph.add_edge(q1, q2, weight=abs_weight, raw_weight=weight)

        if verbose:
            print(f"Network has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

        self._compute_centralities(verbose)
        return self

    def _compute_centralities(self, verbose: bool = True):
        """Compute various centrality measures."""
        if verbose:
            print("Computing centrality measures...")

        scores = pd.DataFrame(index=self.question_cols)

        # Degree centrality (normalized count of connections)
        degree_cent = nx.degree_centrality(self.graph)
        scores['degree_centrality'] = pd.Series(degree_cent)

        # Weighted degree (strength) - sum of edge weights
        strength = {}
        for node in self.graph.nodes():
            strength[node] = sum(d['weight'] for _, _, d in self.graph.edges(node, data=True))
        scores['strength'] = pd.Series(strength)

        # Betweenness centrality (how often on shortest paths)
        # Use weighted version
        try:
            betweenness = nx.betweenness_centrality(self.graph, weight='weight')
            scores['betweenness_centrality'] = pd.Series(betweenness)
        except:
            scores['betweenness_centrality'] = np.nan

        # Eigenvector centrality (connected to important nodes)
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000, weight='weight')
            scores['eigenvector_centrality'] = pd.Series(eigenvector)
        except:
            # May fail if graph is not connected
            scores['eigenvector_centrality'] = np.nan

        # PageRank
        try:
            pagerank = nx.pagerank(self.graph, weight='weight')
            scores['pagerank'] = pd.Series(pagerank)
        except:
            scores['pagerank'] = np.nan

        # Clustering coefficient (how clustered are neighbors)
        clustering = nx.clustering(self.graph, weight='weight')
        scores['clustering_coefficient'] = pd.Series(clustering)

        # Average neighbor degree
        avg_neighbor_degree = nx.average_neighbor_degree(self.graph, weight='weight')
        scores['avg_neighbor_degree'] = pd.Series(avg_neighbor_degree)

        # Composite centrality score
        # Normalize each measure to 0-1 and combine
        for col in ['degree_centrality', 'strength', 'betweenness_centrality',
                    'eigenvector_centrality', 'pagerank']:
            if scores[col].notna().any():
                max_val = scores[col].max()
                if max_val > 0:
                    scores[f'{col}_norm'] = scores[col] / max_val
                else:
                    scores[f'{col}_norm'] = 0

        norm_cols = [c for c in scores.columns if c.endswith('_norm')]
        if norm_cols:
            scores['composite_centrality'] = scores[norm_cols].mean(axis=1)
        else:
            scores['composite_centrality'] = np.nan

        self.scores = scores.sort_values('composite_centrality', ascending=False)

    def get_scores(self) -> pd.DataFrame:
        """Return fundamentalness scores."""
        if self.scores is None:
            raise ValueError("Must call fit() first")
        return self.scores

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Return the correlation matrix."""
        if self.correlation_matrix is None:
            raise ValueError("Must call fit() first")
        return self.correlation_matrix

    def get_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Detect communities in the network using Louvain algorithm.

        Returns dict mapping question -> community_id
        """
        if self.graph is None:
            raise ValueError("Must call fit() first")

        try:
            from networkx.algorithms import community
            communities = community.louvain_communities(self.graph, resolution=resolution, seed=42)
            result = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    result[node] = i
            return result
        except ImportError:
            # Fallback to greedy modularity
            communities = list(nx.community.greedy_modularity_communities(self.graph))
            result = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    result[node] = i
            return result

    def get_network_stats(self) -> Dict:
        """Get summary statistics about the network."""
        if self.graph is None:
            raise ValueError("Must call fit() first")

        stats = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph, weight='weight'),
        }

        # Check if connected
        if nx.is_connected(self.graph):
            stats['is_connected'] = True
            stats['diameter'] = nx.diameter(self.graph)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(self.graph)
        else:
            stats['is_connected'] = False
            stats['n_components'] = nx.number_connected_components(self.graph)

        return stats

    def get_graph(self) -> nx.Graph:
        """Return the networkx graph for custom analysis/visualization."""
        if self.graph is None:
            raise ValueError("Must call fit() first")
        return self.graph
