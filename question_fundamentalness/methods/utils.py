"""
Utility functions for question hierarchy analysis.
Handles missing data, preprocessing, and common operations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
import multiprocessing as mp

DEFAULT_N_JOBS = min(14, mp.cpu_count() - 2)

# Global variables for worker processes
_WORKER_DATA = {}
_WORKER_FUNC = None


def _init_worker(data_dict, func):
    """Initialize worker with shared data."""
    global _WORKER_DATA, _WORKER_FUNC
    _WORKER_DATA = data_dict
    _WORKER_FUNC = func


def _compute_pair(pair):
    """Compute single pair using global worker data."""
    col1, col2, min_samples = pair
    x_full = _WORKER_DATA[col1]
    y_full = _WORKER_DATA[col2]

    mask = ~(np.isnan(x_full) | np.isnan(y_full))
    if mask.sum() < min_samples:
        return col1, col2, None

    try:
        value = _WORKER_FUNC(x_full[mask], y_full[mask])
        return col1, col2, value
    except Exception:
        return col1, col2, None

# Variables to merge (same question on different ballots)
DUPLICATE_PAIRS = [
    ('natspac', 'natspacy'),
    ('natenvir', 'natenviy'),
    ('natheal', 'nathealy'),
    ('natcity', 'natcityy'),
    ('natcrime', 'natcrimy'),
    ('natdrug', 'natdrugy'),
    ('nateduc', 'nateducy'),
    ('natrace', 'natracey'),
    ('natarms', 'natarmsy'),
    ('nataid', 'nataidy'),
    ('natfare', 'natfarey'),
    ('letdie1', 'letdie1y'),
]


def load_and_preprocess(
    filepath: str = 'gss_2021-2024_public_opinion.csv',
    merge_duplicates: bool = True,
    filter_valid_responses: bool = True,
    min_valid_responses: int = 100
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load GSS data and preprocess for analysis.

    Parameters
    ----------
    filepath : str
        Path to the CSV file
    merge_duplicates : bool
        Whether to merge duplicate ballot versions (e.g., natcrime + natcrimy)
    filter_valid_responses : bool
        Whether to convert invalid responses (8, 9, 0, negative) to NaN
    min_valid_responses : int
        Minimum number of valid responses required to keep a variable

    Returns
    -------
    df : pd.DataFrame
        Preprocessed dataframe
    question_cols : List[str]
        List of question column names (excluding id, year, partyid)
    """
    df = pd.read_csv(filepath)

    # Merge duplicate ballot versions
    if merge_duplicates:
        for base, alt in DUPLICATE_PAIRS:
            if base in df.columns and alt in df.columns:
                df[base] = df[base].combine_first(df[alt])
                df = df.drop(columns=[alt])
            elif alt in df.columns and base not in df.columns:
                df = df.rename(columns={alt: base})

    # Get question columns
    # Exclude: id, year_label, partyid (party identification), polviews (self-reported ideology)
    non_question_cols = ['id', 'year_label', 'partyid', 'polviews']
    question_cols = [col for col in df.columns if col not in non_question_cols]

    # Filter valid responses: GSS typically uses 1-7 for valid, 8/9/0 for DK/NA
    if filter_valid_responses:
        for col in question_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Mark invalid responses as NaN
                # Most GSS variables: valid = 1-7, invalid = 0, 8, 9, negative
                df.loc[df[col] <= 0, col] = np.nan
                df.loc[df[col] >= 8, col] = np.nan

    # Filter out variables with too few valid responses
    valid_counts = df[question_cols].notna().sum()
    valid_cols = valid_counts[valid_counts >= min_valid_responses].index.tolist()
    question_cols = [col for col in question_cols if col in valid_cols]

    return df, question_cols


def get_valid_pairs(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    min_samples: int = 50
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get pairwise complete cases for two columns.

    Returns None if insufficient valid pairs.
    """
    mask = df[col1].notna() & df[col2].notna()
    if mask.sum() < min_samples:
        return None
    return df.loc[mask, col1].values, df.loc[mask, col2].values


def get_complete_cases(
    df: pd.DataFrame,
    cols: List[str],
    min_samples: int = 50
) -> Optional[pd.DataFrame]:
    """
    Get complete cases across multiple columns.

    Returns None if insufficient complete cases.
    """
    subset = df[cols].dropna()
    if len(subset) < min_samples:
        return None
    return subset


def discretize_continuous(
    x: np.ndarray,
    n_bins: int = 5,
    strategy: str = 'quantile'
) -> np.ndarray:
    """
    Discretize continuous variable for information-theoretic measures.

    Parameters
    ----------
    x : np.ndarray
        Input array
    n_bins : int
        Number of bins
    strategy : str
        'quantile' for equal-frequency, 'uniform' for equal-width
    """
    if strategy == 'quantile':
        # Equal frequency bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x[~np.isnan(x)], percentiles)
        bins = np.unique(bins)  # Remove duplicate bin edges
    else:
        # Equal width bins
        bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)

    return np.digitize(x, bins[1:-1])


def compute_pairwise_matrix(
    df: pd.DataFrame,
    cols: List[str],
    func: Callable,
    symmetric: bool = True,
    min_samples: int = 50,
    verbose: bool = True,
    n_jobs: int = DEFAULT_N_JOBS
) -> pd.DataFrame:
    """
    Compute pairwise matrix using a custom function (parallel).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cols : List[str]
        Columns to compute pairwise values for
    func : callable
        Function that takes (x, y) arrays and returns a scalar
    symmetric : bool
        Whether the function is symmetric (f(x,y) = f(y,x))
    min_samples : int
        Minimum samples required for valid pair
    verbose : bool
        Whether to print progress
    n_jobs : int
        Number of parallel workers

    Returns
    -------
    pd.DataFrame
        Pairwise matrix
    """
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    # Pre-extract data as numpy arrays
    df_values = {col: df[col].values.astype(float) for col in cols}

    # Generate all pairs to compute
    pairs = []
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if symmetric and j <= i:
                continue
            if col1 == col2:
                continue
            pairs.append((col1, col2, min_samples))

    total = len(pairs)
    if verbose:
        print(f"  Computing {total} pairs using {n_jobs} cores...")

    # Use spawn context (safer for notebooks)
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_jobs, initializer=_init_worker, initargs=(df_values, func)) as pool:
        results = pool.map(_compute_pair, pairs)

    # Fill matrix from results
    for col1, col2, value in results:
        if value is not None:
            matrix.loc[col1, col2] = value
            if symmetric:
                matrix.loc[col2, col1] = value

    # Fill diagonal
    for col in cols:
        valid = df_values[col]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            try:
                matrix.loc[col, col] = func(valid, valid)
            except Exception:
                pass

    if verbose:
        print(f"  Done!")

    return matrix


def compute_pairwise_matrix_sequential(
    df: pd.DataFrame,
    cols: List[str],
    func: Callable,
    symmetric: bool = True,
    min_samples: int = 50,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute pairwise matrix using a custom function (SEQUENTIAL VERSION).
    Use this for debugging or when parallel overhead isn't worth it.
    """
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    total = n * (n - 1) // 2 if symmetric else n * (n - 1)
    computed = 0

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if symmetric and j <= i:
                continue
            if col1 == col2:
                continue

            pair = get_valid_pairs(df, col1, col2, min_samples)
            if pair is not None:
                x, y = pair
                try:
                    value = func(x, y)
                    matrix.loc[col1, col2] = value
                    if symmetric:
                        matrix.loc[col2, col1] = value
                except Exception:
                    pass

            computed += 1
            if verbose and computed % 500 == 0:
                print(f"  Progress: {computed}/{total} pairs computed")

    # Fill diagonal
    for col in cols:
        valid = df[col].dropna().values
        if len(valid) > 0:
            try:
                matrix.loc[col, col] = func(valid, valid)
            except Exception:
                pass

    return matrix


def normalize_scores(scores: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize scores to [0, 1] range.

    Parameters
    ----------
    scores : pd.Series
        Raw scores
    method : str
        'minmax' for min-max scaling, 'rank' for rank-based
    """
    if method == 'minmax':
        return (scores - scores.min()) / (scores.max() - scores.min())
    elif method == 'rank':
        return scores.rank(pct=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_question_metadata(question_cols: List[str]) -> pd.DataFrame:
    """
    Create metadata dataframe for questions with topic assignments.
    """
    # Topic mappings (simplified version)
    TOPIC_MAP = {
        'ab': 'Abortion',
        'nat': 'Government Spending',
        'spk': 'Free Speech Tolerance',
        'col': 'Free Speech Tolerance',
        'lib': 'Free Speech Tolerance',
        'con': 'Institutional Confidence',
        'help': 'Government Role',
        'rac': 'Race',
        'wrk': 'Race',
        'aff': 'Race',
        'disc': 'Discrimination',
        'cap': 'Criminal Justice',
        'gun': 'Criminal Justice',
        'court': 'Criminal Justice',
        'grass': 'Criminal Justice',
        'eq': 'Economic',
        'prayer': 'Religion',
        'sex': 'Social Issues',
        'pill': 'Social Issues',
        'spank': 'Social Issues',
        'let': 'Bioethics',
        'pol': 'Ideology',
    }

    topics = []
    for col in question_cols:
        topic = 'Other'
        for prefix, t in TOPIC_MAP.items():
            if col.startswith(prefix):
                topic = t
                break
        topics.append(topic)

    return pd.DataFrame({
        'variable': question_cols,
        'topic': topics
    })
