#!/usr/bin/env python3
"""
GSS Polarization Analysis with PCA Dimensionality Reduction

This script analyzes LLM polarization on GSS survey questions using PCA
to reduce the dimensionality of the activation space before computing
Mahalanobis distance.

Usage:
    python run_gss_pca.py --model-path /path/to/model --model-name Llama-3.1-8B
    python run_gss_pca.py --help
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pandas as pd
import numpy as np
import torch
import gc
import time
import warnings
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv, LinAlgError
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# Local imports
from config import SYSTEM_MSG_POLITICIAN
from model_utils import load_model, extract_heads_batched, get_model_info
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES

# Default politician CSV path
DEFAULT_POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"


# =============================================================================
# PCA MAHALANOBIS FUNCTIONS
# =============================================================================

def weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted mean along axis 0."""
    weights = weights / weights.sum()
    return np.sum(X * weights[:, np.newaxis], axis=0)


def weighted_median(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted median along axis 0."""
    weights = weights / weights.sum()
    result = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        sorted_idx = np.argsort(X[:, j])
        sorted_vals = X[sorted_idx, j]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)
        median_idx = min(median_idx, len(sorted_vals) - 1)
        result[j] = sorted_vals[median_idx]
    return result


def weighted_cov(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted covariance matrix."""
    weights = weights / weights.sum()
    mean = weighted_mean(X, weights)
    X_centered = X - mean
    cov = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(X)):
        cov += weights[i] * np.outer(X_centered[i], X_centered[i])
    sum_w = weights.sum()
    sum_w2 = (weights ** 2).sum()
    cov = cov / (1 - sum_w2 / (sum_w ** 2))
    return cov


def compute_mahalanobis_pca(
    head_data: np.ndarray,
    group_labels: np.ndarray,
    group_values: tuple = (100, 200),
    n_components: int = 10,
    centroid_method: str = 'mean',
) -> float:
    """
    Compute Mahalanobis distance after PCA dimensionality reduction.

    Args:
        head_data: Activation data for a single head
        group_labels: Party labels for each sample
        group_values: Tuple of (Democrat, Republican) codes
        n_components: Number of PCA components
        centroid_method: 'mean' or 'median' for computing party centroids

    Returns:
        Mahalanobis distance between party centroids
    """
    valid_mask = np.isin(group_labels, group_values)
    X = head_data[valid_mask]
    y = group_labels[valid_mask]
    w = np.ones(len(X))

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X)
    except Exception:
        return 0.0

    try:
        mask1 = y == group_values[0]
        mask2 = y == group_values[1]
        group1 = X_pca[mask1]
        group2 = X_pca[mask2]
        w1 = w[mask1]
        w2 = w[mask2]

        if len(group1) > 5 and len(group2) > 5:
            # Compute centroids using specified method
            if centroid_method == 'median':
                mu_1 = weighted_median(group1, w1)
                mu_2 = weighted_median(group2, w2)
            else:  # default to mean
                mu_1 = weighted_mean(group1, w1)
                mu_2 = weighted_mean(group2, w2)

            # Covariance is always computed using mean-centered data
            cov1 = weighted_cov(group1, w1)
            cov2 = weighted_cov(group2, w2)
            cov_pool = (cov1 + cov2) / 2
            cov_pool += np.eye(cov_pool.shape[0]) * 1e-6

            inv_cov = inv(cov_pool)
            return mahalanobis(mu_1, mu_2, inv_cov)
        else:
            return 0.0
    except (LinAlgError, ValueError):
        return 0.0


def compute_all_head_metrics_pca(
    X_heads: np.ndarray,
    group_labels: np.ndarray,
    group_values: tuple = (100, 200),
    n_components: int = 10,
    centroid_method: str = 'mean',
    n_jobs: int = -1
) -> np.ndarray:
    """Compute PCA-based Mahalanobis for all attention heads in parallel.

    Args:
        X_heads: Activation tensor of shape (N, L, H, D)
        group_labels: Party labels for each sample
        group_values: Tuple of (Democrat, Republican) codes
        n_components: Number of PCA components
        centroid_method: 'mean' or 'median' for computing party centroids
        n_jobs: Number of parallel jobs

    Returns:
        Array of shape (L, H) with Mahalanobis distances per head
    """
    N, L, H, D = X_heads.shape
    flat_heads = [X_heads[:, l, h, :] for l in range(L) for h in range(H)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mahalanobis_pca)(
            head_data, group_labels, group_values, n_components, centroid_method
        )
        for head_data in flat_heads
    )

    return np.array(results).reshape(L, H)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_topics_from_csv(csv_path: str) -> tuple:
    """Load topics from CSV file."""
    df = pd.read_csv(csv_path)
    topics = dict(zip(df['Variable'], df['NaturalLanguageClause']))
    return topics, df


def load_polarization_data(csv_path: str) -> pd.DataFrame:
    """Load GSS survey polarization data."""
    return pd.read_csv(csv_path)


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

def run_analysis_for_topic_pca(
    model, tokenizer,
    topic_name: str,
    topic_desc: str,
    template: str,
    pca_dims: list,
    batch_size: int,
    max_length: int,
    politician_csv: str,
) -> dict:
    """Run PCA analysis for a single topic with both mean and median centroids."""
    t0 = time.time()

    df_politicians = load_politicians(politician_csv)

    prompts = generate_politician_prompts(
        topic_desc,
        df_politicians['fullname'].tolist(),
        template=template
    )
    labels = df_politicians['party_code'].values

    X_heads = extract_heads_batched(
        model, tokenizer,
        prompts,
        SYSTEM_MSG_POLITICIAN,
        batch_size=batch_size,
        max_length=max_length
    )

    results = {'Topic': topic_name}

    # Compute for both mean and median centroid methods
    for centroid_method in ['mean', 'median']:
        suffix = '' if centroid_method == 'mean' else '_median'

        for n_comp in pca_dims:
            grid = compute_all_head_metrics_pca(
                X_heads, labels,
                group_values=(100, 200),
                n_components=n_comp,
                centroid_method=centroid_method
            )
            results[f'Avg_Mahal_PCA{n_comp}{suffix}'] = np.mean(grid)
            results[f'Max_Mahal_PCA{n_comp}{suffix}'] = np.max(grid)

    del X_heads

    elapsed = time.time() - t0
    # Show both mean and median for first PCA dim
    d = pca_dims[0]
    print(f"    {topic_name}: mean={results[f'Avg_Mahal_PCA{d}']:.3f}, median={results[f'Avg_Mahal_PCA{d}_median']:.3f} ({elapsed:.1f}s)", flush=True)

    return results


def run_category_analysis_pca(
    model, tokenizer,
    category_name: str,
    topics: dict,
    template_name: str,
    pca_dims: list,
    batch_size: int,
    max_length: int,
    politician_csv: str,
) -> pd.DataFrame:
    """Run PCA analysis for all topics in a category."""
    template = POLITICIAN_TEMPLATES[template_name]

    print(f"\n{'='*70}")
    print(f"ANALYZING: {category_name.upper()}")
    print(f"{'='*70}")
    print(f"  Topics: {len(topics)}")
    print(f"  Template: '{template_name}'")
    print(f"  PCA dims: {pca_dims}")
    print()

    results = []
    topic_list = list(topics.items())

    for idx, (topic_name, topic_desc) in enumerate(topic_list):
        try:
            result = run_analysis_for_topic_pca(
                model, tokenizer,
                topic_name, topic_desc,
                template=template,
                pca_dims=pca_dims,
                batch_size=batch_size,
                max_length=max_length,
                politician_csv=politician_csv,
            )
            result['category'] = category_name
            results.append(result)

            if (idx + 1) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"    [Memory cleanup at {idx + 1}/{len(topic_list)}]")

        except Exception as e:
            print(f"    ERROR on {topic_name}: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    return pd.DataFrame(results)


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlations_pca(df_llm: pd.DataFrame, df_gss: pd.DataFrame,
                              category: str, pca_dims: list) -> list:
    """Compute correlations for each PCA dimension and centroid method."""
    df_merged = df_llm.merge(
        df_gss[['variable', 'polarization', 'area']].rename(
            columns={'variable': 'Topic', 'polarization': 'GSS_Polarization'}
        ),
        on='Topic', how='inner'
    )

    results = []
    for centroid_method in ['mean', 'median']:
        suffix = '' if centroid_method == 'mean' else '_median'

        for n_comp in pca_dims:
            col = f'Avg_Mahal_PCA{n_comp}{suffix}'
            if col in df_merged.columns:
                pearson = df_merged[col].corr(df_merged['GSS_Polarization'], method='pearson')
                spearman = df_merged[col].corr(df_merged['GSS_Polarization'], method='spearman')
                results.append({
                    'category': category,
                    'pca_dim': n_comp,
                    'centroid_method': centroid_method,
                    'n_topics': len(df_merged),
                    'pearson': pearson,
                    'spearman': spearman,
                    'df_merged': df_merged,
                    'llm_col': col,
                })

    return results


# =============================================================================
# BOOTSTRAP ANALYSIS
# =============================================================================

def _bootstrap_worker(args):
    """Worker function for parallel bootstrap sampling."""
    worker_id, n_iters, sample_size, llm_vals, gss_vals, n_total, top_k, seed = args

    np.random.seed(seed + worker_id)
    results = []

    for i in range(n_iters):
        idx = np.random.choice(n_total, size=sample_size, replace=False)
        r, _ = pearsonr(llm_vals[idx], gss_vals[idx])

        if len(results) < top_k:
            results.append((r, idx.copy()))
            results.sort(key=lambda x: x[0], reverse=True)
        elif r > results[-1][0]:
            results[-1] = (r, idx.copy())
            results.sort(key=lambda x: x[0], reverse=True)

    return results


def bootstrap_sensitivity_parallel(
    df_merged: pd.DataFrame,
    llm_col: str,
    sample_size: int,
    n_iterations: int = 100000,
    top_k: int = 15,
    random_seed: int = 42,
    n_workers: int = None
) -> list:
    """Parallel bootstrap sampling."""
    if n_workers is None:
        n_workers = min(cpu_count(), 20)

    topics = df_merged['Topic'].values
    llm_vals = df_merged[llm_col].values
    gss_vals = df_merged['GSS_Polarization'].values
    n_total = len(topics)

    iters_per_worker = n_iterations // n_workers
    remainder = n_iterations % n_workers

    worker_args = []
    for w in range(n_workers):
        n_iters = iters_per_worker + (1 if w < remainder else 0)
        worker_args.append((w, n_iters, sample_size, llm_vals, gss_vals, n_total, top_k, random_seed))

    print(f"  Running {n_iterations:,} iterations across {n_workers} workers...")
    t0 = time.time()

    with Pool(n_workers) as pool:
        worker_results = pool.map(_bootstrap_worker, worker_args)

    all_results = []
    for wr in worker_results:
        all_results.extend(wr)

    all_results.sort(key=lambda x: x[0], reverse=True)
    top_results = all_results[:top_k]

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    output = []
    for rank, (r, idx_kept) in enumerate(top_results, 1):
        idx_kept_set = set(idx_kept)
        removed_indices = [i for i in range(n_total) if i not in idx_kept_set]
        removed_topics = topics[removed_indices].tolist()

        output.append({
            'rank': rank,
            'pearson_r': r,
            'removed_topics': removed_topics,
            'n_removed': len(removed_topics),
        })

    return output


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='GSS Polarization Analysis with PCA Dimensionality Reduction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model settings
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model (for output files)')
    parser.add_argument('--politician-csv', type=str, default=DEFAULT_POLITICIAN_CSV,
                        help='CSV file with politician data')

    # Data files
    parser.add_argument('--public-topics', type=str, default='../question_lists/public_issues.csv',
                        help='CSV file with public issues topics')
    parser.add_argument('--public-polarization', type=str, default='../data/polarization/public_issues_polarization.csv',
                        help='CSV file with public issues polarization data')
    parser.add_argument('--private-topics', type=str, default='../question_lists/private_life.csv',
                        help='CSV file with private life topics')
    parser.add_argument('--private-polarization', type=str, default='../data/polarization/private_life_polarization.csv',
                        help='CSV file with private life polarization data')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')

    # PCA settings
    parser.add_argument('--pca-dims', type=int, nargs='+', default=[5, 10, 15, 30],
                        help='PCA dimensions to test')

    # Processing settings
    parser.add_argument('--batch-size', type=int, default=80,
                        help='Batch size for model inference')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum sequence length')

    # Bootstrap settings
    parser.add_argument('--bootstrap-public', type=int, default=126,
                        help='Sample size for public issues bootstrap')
    parser.add_argument('--bootstrap-private', type=int, default=70,
                        help='Sample size for private life bootstrap')
    parser.add_argument('--bootstrap-iters-public', type=int, default=1000000,
                        help='Bootstrap iterations for public issues')
    parser.add_argument('--bootstrap-iters-private', type=int, default=100000,
                        help='Bootstrap iterations for private life')
    parser.add_argument('--bootstrap-pca-dim', type=int, default=10,
                        help='PCA dimension to use for bootstrap analysis')

    # Filtering
    parser.add_argument('--min-dem', type=int, default=100,
                        help='Minimum Democrat respondents for GSS filtering')
    parser.add_argument('--min-rep', type=int, default=100,
                        help='Minimum Republican respondents for GSS filtering')
    parser.add_argument('--min-total', type=int, default=200,
                        help='Minimum total respondents for GSS filtering')

    # Excluded topics
    parser.add_argument('--exclude-public', type=str, nargs='*',
                        default=['hubbywk1', 'racdif1', 'racdif2', 'racdif3', 'racdif4',
                                 'workwhts', 'wlthwhts', 'intlwhts'],
                        help='Topics to exclude from public issues')
    parser.add_argument('--exclude-private', type=str, nargs='*',
                        default=['reborn', 'marwht', 'helpful', 'helpfulnv', 'helpfulv'],
                        help='Topics to exclude from private life')

    # Skip options
    parser.add_argument('--skip-bootstrap', action='store_true',
                        help='Skip bootstrap sensitivity analysis')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("GSS POLARIZATION ANALYSIS - PCA")
    print("="*70)
    print(f"\n[Configuration]")
    print(f"  Model: {args.model_name}")
    print(f"  Model path: {args.model_path}")
    print(f"  Politician CSV: {args.politician_csv}")
    print(f"  PCA dimensions: {args.pca_dims}")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Timestamp: {timestamp}")

    # Define categories
    categories = {
        "public_issues": {
            "topics_csv": args.public_topics,
            "polarization_csv": args.public_polarization,
            "template_name": "default",
        },
        "private_life": {
            "topics_csv": args.private_topics,
            "polarization_csv": args.private_polarization,
            "template_name": "opinion",
        },
    }

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    all_topics = {}
    all_polarization = {}

    for cat_name, cat_config in categories.items():
        print(f"\n[{cat_name.upper()}]")

        topics, _ = load_topics_from_csv(cat_config['topics_csv'])
        print(f"  Topics loaded: {len(topics)}")

        pol_df = load_polarization_data(cat_config['polarization_csv'])

        # Filter GSS data
        pol_df = pol_df[
            (pol_df['n_dem'] >= args.min_dem) &
            (pol_df['n_rep'] >= args.min_rep) &
            (pol_df['n_total'] >= args.min_total)
        ]
        all_polarization[cat_name] = pol_df
        print(f"  Polarization data (filtered): {len(pol_df)} variables")

        # Filter topics
        pol_set = set(pol_df['variable'].tolist())
        all_topics[cat_name] = {k: v for k, v in topics.items() if k in pol_set}
        print(f"  Final topics: {len(all_topics[cat_name])}")

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Loading: {args.model_path}")

    model, tokenizer = load_model(args.model_path)
    model_info = get_model_info(model)
    print(f"  Layers: {model_info['num_layers']}, Heads: {model_info['num_heads']}")

    # Run analysis
    category_results = {}

    for cat_name, cat_config in categories.items():
        df_result = run_category_analysis_pca(
            model, tokenizer,
            cat_name,
            all_topics[cat_name],
            cat_config['template_name'],
            pca_dims=args.pca_dims,
            batch_size=args.batch_size,
            max_length=args.max_length,
            politician_csv=args.politician_csv,
        )
        category_results[cat_name] = df_result

        out_path = output_dir / f"df_gss_pca_{cat_name}_{args.model_name}_{timestamp}.pkl"
        df_result.to_pickle(out_path)
        print(f"  Saved: {out_path}")

    # Compute correlations
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    all_correlations = []

    for cat_name in categories.keys():
        df_llm = category_results[cat_name]
        df_gss = all_polarization[cat_name]

        corrs = compute_correlations_pca(df_llm, df_gss, cat_name, args.pca_dims)
        all_correlations.extend(corrs)

        print(f"\n[{cat_name.upper()}]")
        for method in ['mean', 'median']:
            print(f"  Centroid: {method}")
            method_corrs = [c for c in corrs if c['centroid_method'] == method]
            for c in method_corrs:
                print(f"    PCA-{c['pca_dim']:2d}: r={c['pearson']:.4f}, ρ={c['spearman']:.4f} (n={c['n_topics']})")

    # Save correlation summary
    df_corr_summary = pd.DataFrame([{
        'Category': c['category'],
        'PCA_Dim': c['pca_dim'],
        'Centroid_Method': c['centroid_method'],
        'N_Topics': c['n_topics'],
        'Pearson_r': c['pearson'],
        'Spearman_rho': c['spearman'],
    } for c in all_correlations])

    corr_csv = output_dir / f"gss_pca_correlations_{args.model_name}_{timestamp}.csv"
    df_corr_summary.to_csv(corr_csv, index=False)
    print(f"\nSaved: {corr_csv}")

    # Bootstrap analysis
    if not args.skip_bootstrap:
        print("\n" + "="*70)
        print(f"BOOTSTRAP SENSITIVITY ANALYSIS (PCA-{args.bootstrap_pca_dim})")
        print("="*70)

        bootstrap_config = {
            'public_issues': {'sample_size': args.bootstrap_public, 'n_iterations': args.bootstrap_iters_public},
            'private_life': {'sample_size': args.bootstrap_private, 'n_iterations': args.bootstrap_iters_private}
        }

        bootstrap_results = {}

        for centroid_method in ['mean', 'median']:
            suffix = '' if centroid_method == 'mean' else '_median'
            llm_col = f'Avg_Mahal_PCA{args.bootstrap_pca_dim}{suffix}'

            print(f"\n--- Centroid Method: {centroid_method.upper()} ---")

            for corr in all_correlations:
                if corr['pca_dim'] != args.bootstrap_pca_dim:
                    continue
                if corr['centroid_method'] != centroid_method:
                    continue

                cat_name = corr['category']
                df_merged = corr['df_merged']
                config = bootstrap_config[cat_name]

                print(f"\n[{cat_name.upper()}]")
                print(f"  Total: {len(df_merged)}, Sample: {config['sample_size']}")
                print(f"  Original r: {corr['pearson']:.4f}")

                top_samples = bootstrap_sensitivity_parallel(
                    df_merged,
                    llm_col=llm_col,
                    sample_size=config['sample_size'],
                    n_iterations=config['n_iterations'],
                    top_k=15
                )

                bootstrap_results[f'{cat_name}_{centroid_method}'] = top_samples

                print(f"\n  Top 5 samples:")
                for result in top_samples[:5]:
                    print(f"    r={result['pearson_r']:.4f}, removed: {result['removed_topics'][:3]}...")

    # Filtered results
    print("\n" + "="*70)
    print("FILTERED RESULTS (Topic Exclusion)")
    print("="*70)

    excluded_sets = {
        'public_issues': set(args.exclude_public) if args.exclude_public else set(),
        'private_life': set(args.exclude_private) if args.exclude_private else set(),
    }

    filtered_results = []

    for corr in all_correlations:
        cat_name = corr['category']
        pca_dim = corr['pca_dim']
        centroid_method = corr['centroid_method']
        df_merged = corr['df_merged'].copy()
        llm_col = corr['llm_col']

        excluded = excluded_sets[cat_name]
        df_filtered = df_merged[~df_merged['Topic'].isin(excluded)].reset_index(drop=True)

        pearson_filt = df_filtered[llm_col].corr(df_filtered['GSS_Polarization'], method='pearson')
        spearman_filt = df_filtered[llm_col].corr(df_filtered['GSS_Polarization'], method='spearman')

        filtered_results.append({
            'Category': cat_name,
            'PCA_Dim': pca_dim,
            'Centroid_Method': centroid_method,
            'N_Original': len(df_merged),
            'N_Filtered': len(df_filtered),
            'Pearson_Original': corr['pearson'],
            'Pearson_Filtered': pearson_filt,
            'Spearman_Original': corr['spearman'],
            'Spearman_Filtered': spearman_filt,
        })

    df_filtered_summary = pd.DataFrame(filtered_results)

    print(f"\nExcluded (public): {sorted(excluded_sets['public_issues'])}")
    print(f"Excluded (private): {sorted(excluded_sets['private_life'])}")
    print(f"\n{df_filtered_summary.to_string()}")

    filtered_csv = output_dir / f"gss_pca_filtered_{args.model_name}_{timestamp}.csv"
    df_filtered_summary.to_csv(filtered_csv, index=False)
    print(f"\nSaved: {filtered_csv}")

    # Save combined results
    df_combined = pd.concat([df for df in category_results.values()], ignore_index=True)
    df_combined['model'] = args.model_name

    combined_pkl = output_dir / f"df_gss_pca_combined_{args.model_name}_{timestamp}.pkl"
    combined_csv = output_dir / f"df_gss_pca_combined_{args.model_name}_{timestamp}.csv"
    df_combined.to_pickle(combined_pkl)
    df_combined.to_csv(combined_csv, index=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nModel: {args.model_name}")
    print(f"PCA Dimensions: {args.pca_dims}")
    print(f"\nFiles saved to: {output_dir.absolute()}")
    print(f"  - {combined_pkl.name}")
    print(f"  - {combined_csv.name}")
    print(f"  - {corr_csv.name}")
    print(f"  - {filtered_csv.name}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
