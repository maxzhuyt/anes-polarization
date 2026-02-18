"""
Dimensionality Reduction approach to measuring question fundamentalness.

Theory:
- PCA/Factor Analysis finds the main "dimensions" underlying the data
- Questions that load heavily on the first few principal components
  capture the most important sources of variation
- Questions with high "communality" (total variance explained by factors)
  are more central/fundamental to the overall structure

Metrics computed:
1. PC1 loading: Loading on first principal component (absolute value)
2. Total variance explained: Sum of squared loadings across top PCs
3. Communality: Proportion of variance explained by all retained factors
4. Factor complexity: Does question load on one factor or many?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings


class DimensionalityAnalyzer:
    """
    Analyze question fundamentalness using dimensionality reduction.
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
        self.pca = None
        self.fa = None
        self.loadings_pca = None
        self.loadings_fa = None
        self.scores = None

    def _prepare_data(self, min_valid_frac: float = 0.3) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data matrix for dimensionality reduction.

        Handles missing values via imputation for columns with sufficient data.
        """
        # Filter columns with enough valid data
        valid_cols = []
        for col in self.question_cols:
            valid_frac = self.df[col].notna().mean()
            if valid_frac >= min_valid_frac:
                valid_cols.append(col)

        # Get data matrix
        X = self.df[valid_cols].values

        # Impute missing values with column median
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        return X_scaled, valid_cols

    def fit(
        self,
        n_components: int = 10,
        min_valid_frac: float = 0.3,
        verbose: bool = True
    ) -> 'DimensionalityAnalyzer':
        """
        Perform PCA and Factor Analysis.

        Parameters
        ----------
        n_components : int
            Number of components/factors to extract
        min_valid_frac : float
            Minimum fraction of valid responses required for a variable
        verbose : bool
            Print progress
        """
        if verbose:
            print("Preparing data...")

        X, valid_cols = self._prepare_data(min_valid_frac)
        self.valid_cols = valid_cols

        if verbose:
            print(f"Using {len(valid_cols)} variables with sufficient data")
            print(f"Data shape: {X.shape}")

        # Adjust n_components if necessary
        n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)

        # PCA
        if verbose:
            print(f"Running PCA with {n_components} components...")

        self.pca = PCA(n_components=n_components)
        self.pca.fit(X)

        # PCA loadings: correlation between variables and components
        # loadings = eigenvectors * sqrt(eigenvalues)
        self.loadings_pca = pd.DataFrame(
            self.pca.components_.T * np.sqrt(self.pca.explained_variance_),
            index=valid_cols,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        if verbose:
            print(f"PCA variance explained: {self.pca.explained_variance_ratio_[:5].sum():.1%} (first 5 PCs)")

        # Factor Analysis
        if verbose:
            print(f"Running Factor Analysis with {n_components} factors...")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fa = FactorAnalysis(n_components=n_components, random_state=42)
                self.fa.fit(X)

                self.loadings_fa = pd.DataFrame(
                    self.fa.components_.T,
                    index=valid_cols,
                    columns=[f'F{i+1}' for i in range(n_components)]
                )
        except Exception as e:
            if verbose:
                print(f"Factor Analysis failed: {e}")
            self.fa = None
            self.loadings_fa = None

        self._compute_scores(n_components)
        return self

    def _compute_scores(self, n_components: int):
        """Compute fundamentalness scores from loadings."""
        scores = pd.DataFrame(index=self.valid_cols)

        # PCA-based scores
        # Absolute loading on PC1
        scores['pc1_loading'] = self.loadings_pca['PC1'].abs()

        # Total variance captured: sum of squared loadings across components
        scores['total_variance_pca'] = (self.loadings_pca ** 2).sum(axis=1)

        # Max loading: highest loading on any single component
        scores['max_loading_pca'] = self.loadings_pca.abs().max(axis=1)

        # Complexity: does question load on one or many components?
        # Low complexity (loading on few factors) suggests the question is "pure"
        # High complexity (loading on many) suggests it's multidimensional
        loadings_sq = self.loadings_pca ** 2
        loadings_sq_norm = loadings_sq.div(loadings_sq.sum(axis=1), axis=0)
        # Entropy of normalized squared loadings
        scores['loading_entropy'] = -(loadings_sq_norm * np.log(loadings_sq_norm + 1e-10)).sum(axis=1)

        # Factor Analysis scores (if available)
        if self.loadings_fa is not None:
            scores['f1_loading'] = self.loadings_fa['F1'].abs()
            scores['total_variance_fa'] = (self.loadings_fa ** 2).sum(axis=1)
            scores['max_loading_fa'] = self.loadings_fa.abs().max(axis=1)

            # Communality: proportion of variance explained by all factors
            scores['communality'] = (self.loadings_fa ** 2).sum(axis=1)

        # Composite score
        # Questions that load heavily on top PCs and have high communality are fundamental
        score_components = ['pc1_loading', 'total_variance_pca']
        if 'communality' in scores.columns:
            score_components.append('communality')

        for col in score_components:
            max_val = scores[col].max()
            if max_val > 0:
                scores[f'{col}_norm'] = scores[col] / max_val

        norm_cols = [c for c in scores.columns if c.endswith('_norm')]
        scores['composite_dimensionality'] = scores[norm_cols].mean(axis=1)

        # Add questions not in analysis (due to missing data) with NaN scores
        all_scores = pd.DataFrame(index=self.question_cols)
        all_scores = all_scores.join(scores)

        self.scores = all_scores.sort_values('composite_dimensionality', ascending=False)

    def get_scores(self) -> pd.DataFrame:
        """Return fundamentalness scores."""
        if self.scores is None:
            raise ValueError("Must call fit() first")
        return self.scores

    def get_loadings(self, method: str = 'pca') -> pd.DataFrame:
        """
        Return factor loadings.

        Parameters
        ----------
        method : str
            'pca' or 'fa' (factor analysis)
        """
        if method == 'pca':
            if self.loadings_pca is None:
                raise ValueError("Must call fit() first")
            return self.loadings_pca
        else:
            if self.loadings_fa is None:
                raise ValueError("Factor Analysis not available")
            return self.loadings_fa

    def get_variance_explained(self) -> pd.DataFrame:
        """Get variance explained by each component."""
        if self.pca is None:
            raise ValueError("Must call fit() first")

        return pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(self.pca.explained_variance_ratio_))],
            'variance_explained': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_)
        })

    def get_top_loaders(self, component: str = 'PC1', n: int = 10) -> pd.DataFrame:
        """Get questions with highest loadings on a specific component."""
        if self.loadings_pca is None:
            raise ValueError("Must call fit() first")

        if component not in self.loadings_pca.columns:
            raise ValueError(f"Component {component} not found")

        loadings = self.loadings_pca[component].abs().sort_values(ascending=False)
        raw_loadings = self.loadings_pca[component]

        return pd.DataFrame({
            'loading': raw_loadings[loadings.index[:n]],
            'abs_loading': loadings[:n]
        })

    def interpret_components(self, n_components: int = 5, n_vars: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top loading variables for each component to help interpretation.

        Returns dict mapping component name to list of (variable, loading) tuples.
        """
        if self.loadings_pca is None:
            raise ValueError("Must call fit() first")

        result = {}
        for i in range(min(n_components, self.loadings_pca.shape[1])):
            comp_name = f'PC{i+1}'
            loadings = self.loadings_pca[comp_name]
            top_pos = loadings.nlargest(n_vars)
            top_neg = loadings.nsmallest(n_vars)

            result[comp_name] = {
                'positive': [(var, load) for var, load in top_pos.items()],
                'negative': [(var, load) for var, load in top_neg.items()],
                'variance_explained': self.pca.explained_variance_ratio_[i]
            }

        return result
