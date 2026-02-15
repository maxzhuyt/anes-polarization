"""
Metrics utilities for polarization analysis.

This module provides functions to calculate various polarization metrics
from head-wise activations.
"""

import warnings
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv, LinAlgError
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Union
import pandas as pd


def weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted mean along axis 0."""
    weights = weights / weights.sum()
    return np.sum(X * weights[:, np.newaxis], axis=0)


def weighted_cov(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted covariance matrix."""
    weights = weights / weights.sum()
    mean = weighted_mean(X, weights)
    X_centered = X - mean
    # Weighted covariance: sum(w_i * (x_i - mu)(x_i - mu)^T)
    cov = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(X)):
        cov += weights[i] * np.outer(X_centered[i], X_centered[i])
    # Bias correction factor for weighted samples
    sum_w = weights.sum()
    sum_w2 = (weights ** 2).sum()
    cov = cov / (1 - sum_w2 / (sum_w ** 2))
    return cov


def calculate_metrics_for_single_head(
    head_data: np.ndarray,
    group_labels: np.ndarray,
    group_values: tuple = (100, 200),
    sample_weights: np.ndarray = None,
) -> Dict[str, float]:
    """
    Calculate all polarization metrics for a single attention head.

    This function computes multiple metrics measuring separation and structure
    in the activation space between two groups.

    Args:
        head_data: Activation data for one head, shape [N, D]
        group_labels: Group labels for each sample
        group_values: Tuple of (group1_value, group2_value) to filter valid samples
        sample_weights: Optional per-sample weights (e.g., from ANES ideology distribution)

    Returns:
        Dictionary containing:
        - Mahalanobis: Mahalanobis distance between group centroids
        - Davies_Bouldin: Davies-Bouldin clustering index
        - Total_Dispersion: Sum of eigenvalues (total variance)
        - PC1_Ratio: Explained variance ratio of first principal component
        - Intrinsic_Dim: Intrinsic dimensionality via participation ratio
    """
    # Filter to valid groups only
    valid_mask = np.isin(group_labels, group_values)
    X = head_data[valid_mask]
    y = group_labels[valid_mask]

    # Handle weights
    if sample_weights is not None:
        w = sample_weights[valid_mask]
    else:
        w = np.ones(len(X))

    # Centering for PCA/Covariance
    X_centered = X - np.mean(X, axis=0)

    results = {}

    # --- PCA-based metrics ---
    try:
        pca = PCA(n_components=min(10, X.shape[1], X.shape[0] - 1))
        # Suppress warning for heads with near-zero variance (valid edge case)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                    message='invalid value encountered')
            pca.fit(X_centered)
        evals = pca.explained_variance_

        # Total Dispersion (Sum of variance/eigenvalues)
        results['Total_Dispersion'] = np.sum(evals)

        # Explained Variance Ratio of PC1
        results['PC1_Ratio'] = pca.explained_variance_ratio_[0]

        # Intrinsic Dimensionality (Participation Ratio)
        sum_evals = np.sum(evals)
        sum_sq_evals = np.sum(evals**2)
        if sum_sq_evals > 0:
            results['Intrinsic_Dim'] = (sum_evals**2) / sum_sq_evals
        else:
            results['Intrinsic_Dim'] = 0.0

    except Exception:
        results['Total_Dispersion'] = 0.0
        results['PC1_Ratio'] = 0.0
        results['Intrinsic_Dim'] = 0.0

    # --- Cluster metrics ---

    # Davies-Bouldin Index (unweighted - sklearn doesn't support weights)
    try:
        if len(np.unique(y)) > 1:
            results['Davies_Bouldin'] = davies_bouldin_score(X, y)
        else:
            results['Davies_Bouldin'] = 10.0  # Bad score
    except Exception:
        results['Davies_Bouldin'] = 10.0

    # Mahalanobis Distance (with optional weights)
    try:
        mask1 = y == group_values[0]
        mask2 = y == group_values[1]
        group1 = X[mask1]
        group2 = X[mask2]
        w1 = w[mask1]
        w2 = w[mask2]

        if len(group1) > 5 and len(group2) > 5:
            # Weighted means
            mu_1 = weighted_mean(group1, w1)
            mu_2 = weighted_mean(group2, w2)

            # Weighted pooled covariance with regularization
            cov1 = weighted_cov(group1, w1)
            cov2 = weighted_cov(group2, w2)
            cov_pool = (cov1 + cov2) / 2
            cov_pool += np.eye(cov_pool.shape[0]) * 1e-6  # Regularize

            inv_cov = inv(cov_pool)
            results['Mahalanobis'] = mahalanobis(mu_1, mu_2, inv_cov)
        else:
            results['Mahalanobis'] = 0.0
    except (LinAlgError, ValueError):
        results['Mahalanobis'] = 0.0

    return results


def compute_all_head_metrics(
    X_heads: np.ndarray,
    group_labels: np.ndarray,
    group_values: tuple = (100, 200),
    sample_weights: np.ndarray = None,
    n_jobs: int = -1
) -> Dict[str, np.ndarray]:
    """
    Compute metrics for all attention heads in parallel.

    Args:
        X_heads: Activation tensor of shape [N, L, H, D]
        group_labels: Group labels for each sample
        group_values: Tuple of group values to compare
        sample_weights: Optional per-sample weights for weighted Mahalanobis
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Dictionary mapping metric names to [L, H] grids
    """
    N, L, H, D = X_heads.shape

    # Flatten L and H for parallel iteration
    flat_heads = [X_heads[:, l, h, :] for l in range(L) for h in range(H)]

    print(f"  > Computing metrics for {L * H} heads (Parallel)...")

    metrics_flat = Parallel(n_jobs=n_jobs)(
        delayed(calculate_metrics_for_single_head)(head_data, group_labels, group_values, sample_weights)
        for head_data in flat_heads
    )

    # Reshape back to (L, H) grids
    metric_grids = {k: np.zeros((L, H)) for k in metrics_flat[0].keys()}

    idx = 0
    for l in range(L):
        for h in range(H):
            m = metrics_flat[idx]
            for key in m:
                metric_grids[key][l, h] = m[key]
            idx += 1

    return metric_grids


def summarize_metrics(
    metric_grids: Dict[str, np.ndarray],
    topic_name: str
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Create a summary dictionary for a topic's metrics.

    Args:
        metric_grids: Dictionary of [L, H] metric grids
        topic_name: Name of the topic

    Returns:
        Summary dictionary with Avg_, Max_, and Grid_ entries for each metric
    """
    summary = {"Topic": topic_name}

    for key, grid in metric_grids.items():
        summary[f"Avg_{key}"] = np.mean(grid)
        summary[f"Max_{key}"] = np.max(grid)
        summary[f"Grid_{key}"] = grid

    return summary


def calc_center_of_gravity(grid: np.ndarray) -> float:
    """
    Calculate the center of gravity (weighted average layer index) for a 2D grid.

    This measures where in the layer stack the polarization signal is strongest.

    Args:
        grid: 2D array where axis 0 is layer, axis 1 is head

    Returns:
        Float representing the average layer number (0 to L-1)
    """
    if not isinstance(grid, np.ndarray):
        return np.nan

    # Handle invalid values
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)

    L, H = grid.shape
    layer_idx = np.arange(L)
    layer_sum = grid.sum(axis=1)
    total = layer_sum.sum()

    if total == 0:
        return np.nan

    return np.sum(layer_idx * layer_sum) / total


def safe_reciprocal(grid: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal only where values are valid.

    Useful for inverting Davies-Bouldin (where lower is better separation).

    Args:
        grid: Input array

    Returns:
        Reciprocal array with invalid values as nan
    """
    if not isinstance(grid, np.ndarray):
        return np.nan

    result = np.full_like(grid, np.nan, dtype=float)
    mask = np.isfinite(grid) & (grid > 1e-10)
    result[mask] = 1.0 / grid[mask]
    return result


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived metrics to a results DataFrame.

    Args:
        df: DataFrame with Grid_Davies_Bouldin column

    Returns:
        DataFrame with additional derived metrics
    """
    df = df.copy()

    if 'Grid_Davies_Bouldin' in df.columns:
        df['Grid_Davies_Bouldin_Rev'] = df['Grid_Davies_Bouldin'].apply(safe_reciprocal)
        df['Polarization_CoG'] = df['Grid_Davies_Bouldin_Rev'].apply(calc_center_of_gravity)

    return df
