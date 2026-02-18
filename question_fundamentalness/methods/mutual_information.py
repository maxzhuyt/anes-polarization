"""
Mutual Information-based approach to measuring question fundamentalness.

Theory:
- Mutual Information I(X;Y) measures how much knowing X reduces uncertainty about Y
- A "fundamental" question has high MI with many other questions
- We can also compute conditional MI: I(X;Y|Z) to find questions that remain
  informative even after controlling for other variables

Metrics computed:
1. Average MI: Mean mutual information with all other questions
2. Max MI: Maximum MI with any single question
3. MI Breadth: Number of questions with MI above threshold
4. Normalized MI: MI normalized by entropy (uncertainty coefficient)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import entropy

from .utils import get_valid_pairs, compute_pairwise_matrix, DEFAULT_N_JOBS


class MutualInformationAnalyzer:
    """
    Analyze question fundamentalness using mutual information.
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
        self.nmi_matrix = None
        self.scores = None

    def compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two discrete variables."""
        # Ensure integer type for mutual_info_score
        x_int = x.astype(int)
        y_int = y.astype(int)
        return mutual_info_score(x_int, y_int)

    def compute_nmi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute normalized mutual information (0-1 scale)."""
        x_int = x.astype(int)
        y_int = y.astype(int)
        return normalized_mutual_info_score(x_int, y_int)

    def compute_entropy(self, x: np.ndarray) -> float:
        """Compute entropy of a discrete variable."""
        _, counts = np.unique(x[~np.isnan(x)], return_counts=True)
        return entropy(counts, base=2)

    def compute_uncertainty_coefficient(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute uncertainty coefficient U(Y|X) = I(X;Y) / H(Y).

        This measures what fraction of Y's uncertainty is explained by X.
        Asymmetric: U(Y|X) != U(X|Y) in general.
        """
        mi = self.compute_mi(x, y)
        h_y = self.compute_entropy(y)
        if h_y == 0:
            return 0
        return mi / h_y

    def fit(self, verbose: bool = True, n_jobs: int = DEFAULT_N_JOBS) -> 'MutualInformationAnalyzer':
        """
        Compute all pairwise mutual information values.

        Parameters
        ----------
        verbose : bool
            Print progress
        n_jobs : int
            Number of parallel workers
        """
        if verbose:
            print(f"Computing pairwise mutual information matrix ({n_jobs} cores)...")

        self.mi_matrix = compute_pairwise_matrix(
            self.df, self.question_cols, self.compute_mi,
            symmetric=True, verbose=verbose, n_jobs=n_jobs
        )

        if verbose:
            print(f"Computing pairwise normalized MI matrix ({n_jobs} cores)...")

        self.nmi_matrix = compute_pairwise_matrix(
            self.df, self.question_cols, self.compute_nmi,
            symmetric=True, verbose=verbose, n_jobs=n_jobs
        )

        self._compute_scores()
        return self

    def _compute_scores(self):
        """Compute fundamentalness scores from MI matrix."""
        scores = pd.DataFrame(index=self.question_cols)

        # Average MI with all other questions
        scores['avg_mi'] = self.mi_matrix.mean(axis=1)

        # Max MI with any question
        # Set diagonal to NaN first to exclude self-MI
        mi_no_diag = self.mi_matrix.copy()
        np.fill_diagonal(mi_no_diag.values, np.nan)
        scores['max_mi'] = mi_no_diag.max(axis=1)

        # Average normalized MI
        scores['avg_nmi'] = self.nmi_matrix.mean(axis=1)

        # MI breadth: count of questions with NMI > 0.1
        scores['mi_breadth'] = (self.nmi_matrix > 0.1).sum(axis=1) - 1  # subtract self

        # Entropy of each question (for reference)
        entropies = {}
        for col in self.question_cols:
            valid = self.df[col].dropna().values
            if len(valid) > 0:
                entropies[col] = self.compute_entropy(valid)
            else:
                entropies[col] = np.nan
        scores['entropy'] = pd.Series(entropies)

        # Composite score: weighted combination
        # Higher avg_mi and higher breadth = more fundamental
        scores['composite_mi'] = (
            0.5 * (scores['avg_mi'] / scores['avg_mi'].max()) +
            0.3 * (scores['mi_breadth'] / scores['mi_breadth'].max()) +
            0.2 * (scores['entropy'] / scores['entropy'].max())
        )

        self.scores = scores.sort_values('composite_mi', ascending=False)

    def get_scores(self) -> pd.DataFrame:
        """Return fundamentalness scores."""
        if self.scores is None:
            raise ValueError("Must call fit() first")
        return self.scores

    def get_mi_matrix(self) -> pd.DataFrame:
        """Return the full MI matrix."""
        if self.mi_matrix is None:
            raise ValueError("Must call fit() first")
        return self.mi_matrix

    def get_top_pairs(self, n: int = 20) -> pd.DataFrame:
        """Get the top N most mutually informative question pairs."""
        if self.mi_matrix is None:
            raise ValueError("Must call fit() first")

        # Convert matrix to long format
        pairs = []
        for i, col1 in enumerate(self.question_cols):
            for j, col2 in enumerate(self.question_cols):
                if j > i:  # Upper triangle only
                    mi_val = self.mi_matrix.loc[col1, col2]
                    nmi_val = self.nmi_matrix.loc[col1, col2]
                    if not np.isnan(mi_val):
                        pairs.append({
                            'question_1': col1,
                            'question_2': col2,
                            'mi': mi_val,
                            'nmi': nmi_val
                        })

        pairs_df = pd.DataFrame(pairs)
        return pairs_df.nlargest(n, 'mi')

    def get_question_clusters(self, threshold: float = 0.15) -> Dict[str, List[str]]:
        """
        Group questions into clusters based on MI similarity.

        Questions with NMI > threshold are considered related.
        Returns dict mapping each question to its "related" questions.
        """
        if self.nmi_matrix is None:
            raise ValueError("Must call fit() first")

        clusters = {}
        for col in self.question_cols:
            related = self.nmi_matrix.loc[col]
            related = related[related > threshold].index.tolist()
            related = [q for q in related if q != col]
            clusters[col] = related

        return clusters
