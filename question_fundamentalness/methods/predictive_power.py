"""
Predictive Power approach to measuring question fundamentalness.

Theory:
- A "fundamental" question should be able to predict many other questions well
- We use cross-validated out-of-sample prediction to avoid overfitting
- Questions that are good predictors of many others are more fundamental

Metrics computed:
1. Average OOS Accuracy: Mean accuracy predicting all other questions
2. Average OOS R²: Mean explained variance for ordinal predictions
3. Predictive Breadth: Number of questions predicted above chance
4. Asymmetric predictive power: How well X predicts Y vs Y predicts X
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import multiprocessing as mp
import warnings

from .utils import get_valid_pairs, DEFAULT_N_JOBS

# Global variables for worker processes
_PRED_DATA = {}


def _init_pred_worker(data_dict):
    """Initialize worker with shared data."""
    global _PRED_DATA
    _PRED_DATA = data_dict


def _compute_prediction(args):
    """Compute prediction for single pair using global worker data."""
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

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_splits = min(5, len(unique_classes))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            model = LogisticRegression(max_iter=200, solver='lbfgs',
                multi_class='multinomial' if len(unique_classes) > 2 else 'auto', random_state=42)
            acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = max(0, cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='r2').mean())
    except Exception:
        pass

    acc_over_baseline = acc - baseline if not np.isnan(acc) else np.nan
    return predictor, target, acc, acc_over_baseline, r2


class PredictivePowerAnalyzer:
    """
    Analyze question fundamentalness using predictive power.
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
        self.prediction_matrix = None  # prediction_matrix[i,j] = how well q_i predicts q_j
        self.scores = None

    def _compute_predictive_accuracy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Tuple[float, float]:
        """
        Compute cross-validated predictive accuracy of x for y.

        Returns (accuracy, baseline_accuracy) where baseline is majority class.
        """
        # Reshape x for sklearn
        X = x.reshape(-1, 1)

        # Get unique classes in y
        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            return np.nan, np.nan

        # Baseline: majority class accuracy
        _, counts = np.unique(y, return_counts=True)
        baseline = counts.max() / len(y)

        # Use logistic regression for classification
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Stratified K-Fold to handle imbalanced classes
                cv = StratifiedKFold(n_splits=min(n_splits, len(unique_classes)), shuffle=True, random_state=42)

                model = LogisticRegression(
                    max_iter=200,
                    solver='lbfgs',
                    multi_class='multinomial' if len(unique_classes) > 2 else 'auto',
                    random_state=42
                )

                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                return scores.mean(), baseline

        except Exception as e:
            return np.nan, np.nan

    def _compute_r2(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> float:
        """
        Compute cross-validated R² treating y as continuous/ordinal.
        """
        X = x.reshape(-1, 1)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = Ridge(alpha=1.0)
                scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')
                return max(0, scores.mean())  # Clip negative R² to 0

        except Exception:
            return np.nan

    def fit(self, min_samples: int = 100, verbose: bool = True, n_jobs: int = DEFAULT_N_JOBS) -> 'PredictivePowerAnalyzer':
        """
        Compute pairwise predictive power matrix (parallel).

        Note: This matrix is ASYMMETRIC - prediction_matrix[i,j] is how well
        question i predicts question j, which may differ from [j,i].

        Parameters
        ----------
        min_samples : int
            Minimum samples required for valid prediction
        verbose : bool
            Print progress
        n_jobs : int
            Number of parallel workers
        """
        n = len(self.question_cols)

        # Initialize matrices
        acc_matrix = pd.DataFrame(np.nan, index=self.question_cols, columns=self.question_cols)
        acc_over_baseline = pd.DataFrame(np.nan, index=self.question_cols, columns=self.question_cols)
        r2_matrix = pd.DataFrame(np.nan, index=self.question_cols, columns=self.question_cols)

        # Pre-extract data as numpy arrays
        df_values = {col: self.df[col].values.astype(float) for col in self.question_cols}

        # Generate all pairs (asymmetric, so we need all i,j where i != j)
        pairs = [(pred, targ, min_samples) for pred in self.question_cols
                 for targ in self.question_cols if pred != targ]

        total = len(pairs)

        if verbose:
            print(f"Computing predictive power matrix ({total} pairs) using {n_jobs} cores...")

        # Use spawn context (safer for notebooks)
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_jobs, initializer=_init_pred_worker, initargs=(df_values,)) as pool:
            all_results = pool.map(_compute_prediction, pairs)

        # Fill matrices from results
        for predictor, target, acc, acc_over_base, r2 in all_results:
            if not np.isnan(acc):
                acc_matrix.loc[predictor, target] = acc
                acc_over_baseline.loc[predictor, target] = acc_over_base
            if not np.isnan(r2):
                r2_matrix.loc[predictor, target] = r2

        self.acc_matrix = acc_matrix
        self.acc_over_baseline = acc_over_baseline
        self.r2_matrix = r2_matrix

        if verbose:
            print(f"  Done!")

        self._compute_scores()
        return self

    def _compute_scores(self):
        """Compute fundamentalness scores from prediction matrices."""
        scores = pd.DataFrame(index=self.question_cols)

        # How well does this question predict others (row mean)
        scores['avg_predictive_acc'] = self.acc_matrix.mean(axis=1)
        scores['avg_acc_over_baseline'] = self.acc_over_baseline.mean(axis=1)
        scores['avg_predictive_r2'] = self.r2_matrix.mean(axis=1)

        # How well is this question predicted by others (column mean)
        scores['avg_predicted_acc'] = self.acc_matrix.mean(axis=0)
        scores['avg_predicted_r2'] = self.r2_matrix.mean(axis=0)

        # Predictive breadth: how many questions can this predict above baseline?
        scores['predictive_breadth'] = (self.acc_over_baseline > 0.05).sum(axis=1)

        # Asymmetry: questions that predict well but aren't easily predicted
        # are potentially more "fundamental" (they're upstream)
        scores['predictive_asymmetry'] = (
            scores['avg_predictive_r2'] - scores['avg_predicted_r2']
        )

        # Composite score
        # High predictive power + high breadth + positive asymmetry = fundamental
        scores['composite_predictive'] = (
            0.4 * (scores['avg_acc_over_baseline'] / scores['avg_acc_over_baseline'].max()).clip(0, 1) +
            0.3 * (scores['avg_predictive_r2'] / scores['avg_predictive_r2'].max()).clip(0, 1) +
            0.2 * (scores['predictive_breadth'] / scores['predictive_breadth'].max()) +
            0.1 * ((scores['predictive_asymmetry'] - scores['predictive_asymmetry'].min()) /
                   (scores['predictive_asymmetry'].max() - scores['predictive_asymmetry'].min()))
        )

        self.scores = scores.sort_values('composite_predictive', ascending=False)

    def get_scores(self) -> pd.DataFrame:
        """Return fundamentalness scores."""
        if self.scores is None:
            raise ValueError("Must call fit() first")
        return self.scores

    def get_prediction_asymmetries(self) -> pd.DataFrame:
        """
        Find question pairs with high predictive asymmetry.

        If X predicts Y well but Y doesn't predict X well,
        X may be more "fundamental" than Y.
        """
        if self.r2_matrix is None:
            raise ValueError("Must call fit() first")

        asymmetries = []
        for i, q1 in enumerate(self.question_cols):
            for j, q2 in enumerate(self.question_cols):
                if i >= j:
                    continue

                r2_12 = self.r2_matrix.loc[q1, q2]
                r2_21 = self.r2_matrix.loc[q2, q1]

                if np.isnan(r2_12) or np.isnan(r2_21):
                    continue

                asymmetry = r2_12 - r2_21
                if abs(asymmetry) > 0.02:  # Meaningful asymmetry
                    asymmetries.append({
                        'predictor': q1 if asymmetry > 0 else q2,
                        'predicted': q2 if asymmetry > 0 else q1,
                        'r2_forward': max(r2_12, r2_21),
                        'r2_backward': min(r2_12, r2_21),
                        'asymmetry': abs(asymmetry)
                    })

        return pd.DataFrame(asymmetries).sort_values('asymmetry', ascending=False)

    def get_best_predictors_for(self, target: str, n: int = 10) -> pd.DataFrame:
        """Get the best predictor questions for a given target question."""
        if self.r2_matrix is None:
            raise ValueError("Must call fit() first")

        predictors = self.r2_matrix[target].dropna().sort_values(ascending=False)
        return predictors.head(n).to_frame('r2')
