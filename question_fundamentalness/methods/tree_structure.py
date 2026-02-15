"""
Tree Structure approach to measuring question fundamentalness.

Theory:
- The Chow-Liu algorithm finds the optimal tree-structured approximation
  to a joint probability distribution by finding the maximum-weight spanning
  tree of mutual information values
- Questions closer to the "root" of this tree are more fundamental
- We can also use hierarchical clustering to build a dendrogram

Metrics computed:
1. Tree depth: Distance from root in Chow-Liu tree
2. Subtree size: Number of descendants in the tree
3. Tree centrality: Centrality in the tree structure
4. Dendrogram height: At what height does this question merge?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import networkx as nx

from .utils import get_valid_pairs, compute_pairwise_matrix, DEFAULT_N_JOBS


class TreeStructureAnalyzer:
    """
    Analyze question fundamentalness using tree structures.
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
        self.mi_matrix = None
        self.chow_liu_tree = None
        self.linkage_matrix = None
        self.scores = None

    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information."""
        return mutual_info_score(x.astype(int), y.astype(int))

    def _build_mi_matrix(self, min_samples: int = 50, verbose: bool = True, n_jobs: int = DEFAULT_N_JOBS) -> pd.DataFrame:
        """Build pairwise MI matrix (PARALLEL VERSION)."""
        if verbose:
            print(f"Computing MI matrix using {n_jobs} cores...")

        mi_matrix = compute_pairwise_matrix(
            self.df, self.question_cols, self._compute_mi,
            symmetric=True, min_samples=min_samples, verbose=verbose, n_jobs=n_jobs
        )

        # Replace NaN with 0 for MI (no relationship = 0 MI)
        mi_matrix = mi_matrix.fillna(0)

        return mi_matrix

    def _build_chow_liu_tree(self, verbose: bool = True) -> nx.Graph:
        """
        Build Chow-Liu tree (maximum spanning tree of MI values).

        The Chow-Liu tree is the optimal tree-structured approximation
        to the joint distribution.
        """
        if verbose:
            print("Building Chow-Liu tree (maximum spanning tree of MI)...")

        # Create complete graph with MI as edge weights
        G = nx.Graph()
        G.add_nodes_from(self.question_cols)

        for i, col1 in enumerate(self.question_cols):
            for j, col2 in enumerate(self.question_cols):
                if j <= i:
                    continue
                mi = self.mi_matrix.loc[col1, col2]
                if mi > 0:  # Only add edges with positive MI
                    G.add_edge(col1, col2, weight=mi)

        # Find maximum spanning tree
        # networkx finds minimum, so we negate weights
        for u, v in G.edges():
            G[u][v]['neg_weight'] = -G[u][v]['weight']

        mst = nx.minimum_spanning_tree(G, weight='neg_weight')

        # Restore original weights
        for u, v in mst.edges():
            mst[u][v]['weight'] = -mst[u][v]['neg_weight']
            del mst[u][v]['neg_weight']

        if verbose:
            print(f"Chow-Liu tree has {mst.number_of_edges()} edges")

        return mst

    def _compute_tree_metrics(self, verbose: bool = True) -> pd.DataFrame:
        """Compute tree-based fundamentalness metrics."""
        if verbose:
            print("Computing tree-based metrics...")

        tree = self.chow_liu_tree
        scores = pd.DataFrame(index=self.question_cols)

        # Find the best root: node with highest total MI (most connected)
        total_mi = self.mi_matrix.sum(axis=1)
        root = total_mi.idxmax()
        self.tree_root = root

        if verbose:
            print(f"Selected root: {root} (highest total MI)")

        # Compute depth from root
        depths = nx.single_source_shortest_path_length(tree, root)
        scores['tree_depth'] = pd.Series(depths)

        # Compute subtree sizes (number of descendants)
        # Convert to directed tree rooted at our chosen root
        directed_tree = nx.bfs_tree(tree, root)

        subtree_sizes = {}
        for node in self.question_cols:
            if node in directed_tree:
                # Count all descendants
                descendants = nx.descendants(directed_tree, node)
                subtree_sizes[node] = len(descendants)
            else:
                subtree_sizes[node] = 0
        scores['subtree_size'] = pd.Series(subtree_sizes)

        # Tree centrality (betweenness in the tree)
        betweenness = nx.betweenness_centrality(tree)
        scores['tree_betweenness'] = pd.Series(betweenness)

        # Degree in tree
        degrees = dict(tree.degree())
        scores['tree_degree'] = pd.Series(degrees)

        # Edge weight sum (total MI with tree neighbors)
        neighbor_mi = {}
        for node in tree.nodes():
            neighbor_mi[node] = sum(tree[node][neighbor]['weight']
                                    for neighbor in tree.neighbors(node))
        scores['neighbor_mi_sum'] = pd.Series(neighbor_mi)

        return scores

    def _compute_hierarchy_metrics(self, verbose: bool = True) -> pd.DataFrame:
        """Compute hierarchical clustering-based metrics."""
        if verbose:
            print("Computing hierarchical clustering metrics...")

        scores = pd.DataFrame(index=self.question_cols)

        # Convert MI to distance (1 - normalized MI)
        # Higher MI = smaller distance
        mi_max = self.mi_matrix.max().max()
        if mi_max > 0:
            dist_matrix = 1 - (self.mi_matrix / mi_max)
        else:
            dist_matrix = pd.DataFrame(1.0, index=self.question_cols, columns=self.question_cols)

        # Set diagonal to 0
        np.fill_diagonal(dist_matrix.values, 0)

        # Handle any NaN values
        dist_matrix = dist_matrix.fillna(1.0)

        # Convert to condensed form for scipy
        try:
            condensed = squareform(dist_matrix.values)

            # Perform hierarchical clustering
            self.linkage_matrix = linkage(condensed, method='average')

            # Get merge heights for each variable
            # The height at which a variable first merges indicates how
            # "unique" it is - lower = more similar to others = more central
            n = len(self.question_cols)
            first_merge_height = {}

            for i, col in enumerate(self.question_cols):
                # Find the first merge involving this variable
                for step, (c1, c2, height, _) in enumerate(self.linkage_matrix):
                    c1, c2 = int(c1), int(c2)
                    if c1 == i or c2 == i:
                        first_merge_height[col] = height
                        break
                    # Check if this variable is in a previously formed cluster
                    if c1 >= n:  # c1 is a cluster
                        cluster_idx = c1 - n
                        # This gets complicated - simplify by using cophenetic distance
                        pass

            # Use cophenetic distances instead (more robust)
            from scipy.cluster.hierarchy import cophenet
            from scipy.spatial.distance import pdist

            coph_dists, _ = cophenet(self.linkage_matrix, condensed)

            # Average cophenetic distance for each variable
            coph_matrix = squareform(coph_dists)
            avg_coph_dist = pd.Series(coph_matrix.mean(axis=1), index=self.question_cols)
            scores['avg_cophenetic_dist'] = avg_coph_dist

            # Lower cophenetic distance = merges earlier = more central
            scores['hierarchy_centrality'] = 1 - (avg_coph_dist / avg_coph_dist.max())

        except Exception as e:
            if verbose:
                print(f"Hierarchical clustering failed: {e}")
            scores['avg_cophenetic_dist'] = np.nan
            scores['hierarchy_centrality'] = np.nan

        return scores

    def fit(self, min_samples: int = 50, verbose: bool = True, n_jobs: int = DEFAULT_N_JOBS) -> 'TreeStructureAnalyzer':
        """
        Build tree structures and compute metrics.

        Parameters
        ----------
        min_samples : int
            Minimum samples for valid pair
        verbose : bool
            Print progress
        n_jobs : int
            Number of parallel jobs (default: 14)
        """
        # Build MI matrix
        self.mi_matrix = self._build_mi_matrix(min_samples, verbose, n_jobs)

        # Build Chow-Liu tree
        self.chow_liu_tree = self._build_chow_liu_tree(verbose)

        # Compute tree metrics
        tree_scores = self._compute_tree_metrics(verbose)

        # Compute hierarchy metrics
        hier_scores = self._compute_hierarchy_metrics(verbose)

        # Combine scores
        self.scores = tree_scores.join(hier_scores)

        # Composite score
        # Lower depth + higher subtree size + higher centrality = more fundamental
        self.scores['depth_score'] = 1 - (self.scores['tree_depth'] / self.scores['tree_depth'].max())
        self.scores['subtree_score'] = self.scores['subtree_size'] / self.scores['subtree_size'].max()

        self.scores['composite_tree'] = (
            0.3 * self.scores['depth_score'].fillna(0) +
            0.2 * self.scores['subtree_score'].fillna(0) +
            0.2 * (self.scores['tree_betweenness'] / self.scores['tree_betweenness'].max()).fillna(0) +
            0.15 * (self.scores['neighbor_mi_sum'] / self.scores['neighbor_mi_sum'].max()).fillna(0) +
            0.15 * self.scores['hierarchy_centrality'].fillna(0)
        )

        self.scores = self.scores.sort_values('composite_tree', ascending=False)
        return self

    def get_scores(self) -> pd.DataFrame:
        """Return fundamentalness scores."""
        if self.scores is None:
            raise ValueError("Must call fit() first")
        return self.scores

    def get_chow_liu_tree(self) -> nx.Graph:
        """Return the Chow-Liu tree."""
        if self.chow_liu_tree is None:
            raise ValueError("Must call fit() first")
        return self.chow_liu_tree

    def get_tree_edges(self) -> pd.DataFrame:
        """Get edges of the Chow-Liu tree with their MI weights."""
        if self.chow_liu_tree is None:
            raise ValueError("Must call fit() first")

        edges = []
        for u, v, data in self.chow_liu_tree.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'mi': data['weight']
            })

        return pd.DataFrame(edges).sort_values('mi', ascending=False)

    def get_tree_path(self, source: str, target: str) -> List[str]:
        """Get the path between two questions in the Chow-Liu tree."""
        if self.chow_liu_tree is None:
            raise ValueError("Must call fit() first")

        try:
            return nx.shortest_path(self.chow_liu_tree, source, target)
        except nx.NetworkXNoPath:
            return []

    def get_subtree(self, node: str) -> List[str]:
        """Get all descendants of a node in the rooted tree."""
        if self.chow_liu_tree is None:
            raise ValueError("Must call fit() first")

        directed_tree = nx.bfs_tree(self.chow_liu_tree, self.tree_root)
        if node in directed_tree:
            return list(nx.descendants(directed_tree, node))
        return []
