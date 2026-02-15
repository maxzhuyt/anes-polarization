"""
Main pipeline for LLM polarization analysis.

This script extracts activation patterns from an LLM when prompted about
various political topics, computes polarization metrics, and outputs
results as a table for comparison with survey data.

Usage:
    python pipeline.py [--politician] [--ideology] [--both]

Output:
    - df_llm_politician_{timestamp}.pkl: Results from politician prompts
    - df_llm_ideology_{timestamp}.pkl: Results from ideology prompts
"""

import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np

from config import TOPICS, DEFAULT_MODEL_PATH, DEFAULT_BATCH_SIZE, NOMINATE_CSV
from model_utils import load_model, extract_heads_batched
from metrics_utils import compute_all_head_metrics, summarize_metrics, add_derived_metrics
from prompt_utils import (
    create_politician_prompt_set,
    create_ideology_prompt_set,
    compute_anes_weights,
    PromptSet,
)


def run_analysis_for_topic(
    model,
    tokenizer,
    topic_name: str,
    topic_desc: str,
    prompt_set: PromptSet,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_weights: np.ndarray = None,
) -> dict:
    """
    Run the full analysis pipeline for a single topic.

    Args:
        model: Loaded transformer model
        tokenizer: Tokenizer
        topic_name: Name of the topic
        topic_desc: Description of the topic
        prompt_set: PromptSet containing prompts and labels
        batch_size: Batch size for extraction
        sample_weights: Optional per-sample weights for weighted Mahalanobis

    Returns:
        Summary dictionary with all metrics
    """
    t0 = time.time()
    print(f"\n--- Topic: {topic_name} ---")

    # Extract head activations
    X_heads = extract_heads_batched(
        model, tokenizer,
        prompt_set.prompts,
        prompt_set.system_msg,
        batch_size=batch_size
    )

    # Compute metrics
    # For politician prompts: labels are party codes (100, 200)
    # For ideology prompts: labels are mapped to party codes, with optional ANES weights
    metric_grids = compute_all_head_metrics(
        X_heads,
        prompt_set.labels,
        group_values=(100, 200),
        sample_weights=sample_weights,
    )

    # Summarize
    summary = summarize_metrics(metric_grids, topic_name)

    print(f"  > Done in {time.time() - t0:.1f}s. Avg Mahalanobis: {summary['Avg_Mahalanobis']:.4f}")

    return summary


def run_politician_pipeline(
    model,
    tokenizer,
    topics: dict = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    template: str = None
) -> pd.DataFrame:
    """
    Run the analysis pipeline using politician prompts.

    Args:
        model: Loaded transformer model
        tokenizer: Tokenizer
        topics: Dictionary of topics to analyze (default: TOPICS)
        batch_size: Batch size for extraction
        template: Optional custom prompt template

    Returns:
        DataFrame with results for all topics
    """
    if topics is None:
        topics = TOPICS

    results = []

    print("\n" + "="*60)
    print("POLITICIAN PROMPT PIPELINE")
    print("="*60)

    for topic_name, topic_desc in topics.items():
        prompt_set = create_politician_prompt_set(
            topic_desc,
            csv_path=NOMINATE_CSV,
            template=template
        )

        summary = run_analysis_for_topic(
            model, tokenizer,
            topic_name, topic_desc,
            prompt_set,
            batch_size
        )
        summary['prompt_type'] = 'politician'
        results.append(summary)

    df = pd.DataFrame(results)
    df = add_derived_metrics(df)
    return df


def run_ideology_pipeline(
    model,
    tokenizer,
    topics: dict = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_per_level: int = 20,
    template: str = None,
    use_anes_weights: bool = True,
) -> pd.DataFrame:
    """
    Run the analysis pipeline using generic ideological label prompts.

    Args:
        model: Loaded transformer model
        tokenizer: Tokenizer
        topics: Dictionary of topics to analyze (default: TOPICS)
        batch_size: Batch size for extraction
        n_per_level: Number of prompt templates per ideology level (uniform sampling)
        template: Optional custom prompt template
        use_anes_weights: If True, apply ANES weights during Mahalanobis calculation

    Returns:
        DataFrame with results for all topics
    """
    if topics is None:
        topics = TOPICS

    results = []

    print("\n" + "="*60)
    print("IDEOLOGY PROMPT PIPELINE")
    print(f"  Templates per level: {n_per_level}")
    print(f"  ANES-weighted Mahalanobis: {use_anes_weights}")
    print("="*60)

    for topic_name, topic_desc in topics.items():
        prompt_set = create_ideology_prompt_set(
            topic_desc,
            n_per_level=n_per_level,
            template=template,
            exclude_moderates=True,
        )

        # Compute ANES weights if requested
        sample_weights = None
        if use_anes_weights and prompt_set.ideology_scores is not None:
            sample_weights = compute_anes_weights(prompt_set.ideology_scores)

        summary = run_analysis_for_topic(
            model, tokenizer,
            topic_name, topic_desc,
            prompt_set,
            batch_size,
            sample_weights=sample_weights,
        )
        summary['prompt_type'] = 'ideology'
        results.append(summary)

    df = pd.DataFrame(results)
    df = add_derived_metrics(df)
    return df


def main():
    parser = argparse.ArgumentParser(description='LLM Polarization Analysis Pipeline')
    parser.add_argument('--politician', action='store_true', help='Run politician prompt analysis')
    parser.add_argument('--ideology', action='store_true', help='Run ideology prompt analysis')
    parser.add_argument('--both', action='store_true', help='Run both analyses')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Model path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--n-per-level', type=int, default=20,
                        help='Number of prompt templates per ideology level (uniform sampling)')
    parser.add_argument('--no-anes-weights', action='store_true',
                        help='Disable ANES weighting in Mahalanobis calculation')

    args = parser.parse_args()

    # Default to both if nothing specified
    if not (args.politician or args.ideology or args.both):
        args.both = True

    run_politician = args.politician or args.both
    run_ideology = args.ideology or args.both

    # Load model
    model, tokenizer = load_model(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run analyses
    if run_politician:
        df_politician = run_politician_pipeline(model, tokenizer, batch_size=args.batch_size)
        out_path = f"df_llm_politician_{timestamp}.pkl"
        df_politician.to_pickle(out_path)
        print(f"\nSaved politician results to: {out_path}")

    if run_ideology:
        df_ideology = run_ideology_pipeline(
            model, tokenizer,
            batch_size=args.batch_size,
            n_per_level=args.n_per_level,
            use_anes_weights=not args.no_anes_weights,
        )
        out_path = f"df_llm_ideology_{timestamp}.pkl"
        df_ideology.to_pickle(out_path)
        print(f"\nSaved ideology results to: {out_path}")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
