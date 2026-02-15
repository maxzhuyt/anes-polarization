"""
Common utilities for analysis scripts.
Handles data loading, filtering, and saving results.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.utils import load_and_preprocess, DUPLICATE_PAIRS


def load_data_with_year_filter(
    filepath: str = 'gss_2021-2024_public_opinion.csv',
    min_years: int = 2,
    min_valid_per_year: int = 50
) -> tuple:
    """
    Load data and filter to questions appearing in at least min_years.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    min_years : int
        Minimum number of years a question must appear in (default: 2)
    min_valid_per_year : int
        Minimum valid responses per year to count as "appearing" (default: 50)

    Returns
    -------
    df : pd.DataFrame
        Full dataframe
    question_cols : list
        Filtered question columns
    year_availability : dict
        Dict mapping question -> list of years it appears in
    """
    # Load with standard preprocessing
    df, all_question_cols = load_and_preprocess(
        filepath,
        merge_duplicates=True,
        filter_valid_responses=True,
        min_valid_responses=100
    )

    # Get years in data
    years = sorted(df['year_label'].unique())

    # Check which questions have data in which years
    year_availability = {}
    for col in all_question_cols:
        availability = []
        for year in years:
            year_data = df[df['year_label'] == year][col]
            valid_count = year_data.notna().sum()
            if valid_count >= min_valid_per_year:
                availability.append(int(year))
        year_availability[col] = availability

    # Filter to questions in at least min_years
    filtered_cols = [
        q for q, yrs in year_availability.items()
        if len(yrs) >= min_years
    ]

    print(f"Loaded {len(df)} respondents across years: {years}")
    print(f"Total questions: {len(all_question_cols)}")
    print(f"Questions in >= {min_years} years: {len(filtered_cols)}")

    return df, filtered_cols, year_availability


def save_results(results_dict: dict, output_prefix: str):
    """
    Save analysis results to CSV files.

    Parameters
    ----------
    results_dict : dict
        Dict with keys like 'scores', 'matrix', etc. and DataFrame values
    output_prefix : str
        Prefix for output files (e.g., 'results/mi' -> 'results/mi_scores.csv')
    """
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)

    for name, data in results_dict.items():
        if isinstance(data, pd.DataFrame):
            filepath = f"{output_prefix}_{name}.csv"
            data.to_csv(filepath)
            print(f"Saved: {filepath}")
        elif isinstance(data, dict):
            filepath = f"{output_prefix}_{name}.csv"
            pd.DataFrame.from_dict(data, orient='index', columns=['value']).to_csv(filepath)
            print(f"Saved: {filepath}")
