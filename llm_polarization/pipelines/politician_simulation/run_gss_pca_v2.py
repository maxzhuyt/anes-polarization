"""
GSS Politician Simulation V2: Prompt Framing + Layer Selection

0A: Does prompt type (rhetorical/stance/survey) affect Mahalanobis-GSS correlation?
0B: Does restricting to middle 10% of layers improve correlation?

Supports three topic sets (--topics flag):
  public   — 126 filtered public issues (default, existing behaviour)
  private  — all ~99 private-life topics (no exclusion filter, incl. previously excluded)
  all      — all public + all private (no exclusion filters)

Model families run as 4 groups for parallel ssd-gpu submission:
    python run_gss_pca_v2.py --group 1 [--topics public|private|all]
    python run_gss_pca_v2.py --group 2
    python run_gss_pca_v2.py --group 3
    python run_gss_pca_v2.py --group 4
"""

import os
import sys
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gc
import argparse
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians
from config import SYSTEM_MSG_POLITICIAN
from run_gss_pca import compute_all_head_metrics_pca

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "politician_sim_v2"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
QUESTION_LISTS_DIR = Path("/project/jevans/maxzhuyt/gss_polarization/question_lists")
MODELS_DIR = "/project/jevans/maxzhuyt/models"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Topics filtered out of the *public* run for measurement quality reasons.
# Only applied when --topics public; private and all modes skip all filters.
EXCLUDED_PUBLIC = [
    "hubbywk1", "racdif1", "racdif2", "racdif3", "racdif4",
    "workwhts", "wlthwhts", "intlwhts",
]

# Model groups for GPU splitting
MODEL_GROUPS = {
    1: {  # GPU 1: Gemma-2-9b (2 models, 9B each)
        "Gemma-2-9b_base": {
            "path": f"{MODELS_DIR}/gemma-2-9b",
            "type": "base", "batch_size": 100, "family": "Gemma-2-9b",
        },
        "Gemma-2-9b_instruct": {
            "path": f"{MODELS_DIR}/gemma-2-9b-it",
            "type": "instruct", "batch_size": 100, "family": "Gemma-2-9b",
        },
    },
    2: {  # GPU 2: Llama-3.1-8B (2 models, 8B each)
        "Llama-3.1-8B_base": {
            "path": f"{MODELS_DIR}/Meta-Llama-3.1-8B",
            "type": "base", "batch_size": 110, "family": "Llama-3.1-8B",
        },
        "Llama-3.1-8B_instruct": {
            "path": f"{MODELS_DIR}/Meta-Llama-3.1-8B-Instruct",
            "type": "instruct", "batch_size": 110, "family": "Llama-3.1-8B",
        },
    },
    3: {  # GPU 3: Qwen3-4B (3 models, 4B each)
        "Qwen3-4B_base": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Base",
            "type": "base", "batch_size": 180, "family": "Qwen3-4B",
        },
        "Qwen3-4B_instruct": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Instruct-2507",
            "type": "instruct", "batch_size": 180, "family": "Qwen3-4B",
        },
        "Qwen3-4B_reasoning": {
            "path": f"{MODELS_DIR}/Qwen3-4B-Thinking-2507",
            "type": "reasoning", "batch_size": 180, "family": "Qwen3-4B",
        },
    },
    4: {  # GPU 4: SmolLM3-3B (3 models, 3B each)
        "SmolLM3-3B_base": {
            "path": f"{MODELS_DIR}/SmolLM3-3B-Base",
            "type": "base", "batch_size": 200, "family": "SmolLM3-3B",
        },
        "SmolLM3-3B_instruct": {
            "path": f"{MODELS_DIR}/SmolLM3-3B",
            "type": "instruct", "batch_size": 200, "family": "SmolLM3-3B",
            "system_override": "/no_think",
        },
        "SmolLM3-3B_reasoning": {
            "path": f"{MODELS_DIR}/SmolLM3-3B",
            "type": "reasoning", "batch_size": 200, "family": "SmolLM3-3B",
            "system_override": "/think",
        },
    },
}

CONDITIONS = ["rhetorical", "stance", "survey"]


# =============================================================================
# Helper functions
# =============================================================================

def set_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_polarization_data(csv_path=None) -> pd.DataFrame:
    """Load aggregated GSS polarization statistics from CSV."""
    if csv_path is None:
        public_pol = pd.read_csv(DATA_DIR / "polarization" / "public_issues_polarization.csv")
        private_pol = pd.read_csv(DATA_DIR / "polarization" / "private_life_polarization.csv")
        df = pd.concat([public_pol, private_pol], ignore_index=True)
    else:
        df = pd.read_csv(csv_path)
    return df


# =============================================================================
# Topic loading
# =============================================================================

def load_filtered_public_topics() -> Dict[str, str]:
    """Load filtered public issue topics (EXCLUDED_PUBLIC removed)."""
    csv_path = QUESTION_LISTS_DIR / "public_issues.csv"
    df = pd.read_csv(csv_path)
    topics = {}
    for _, row in df.iterrows():
        var = str(row['Variable']).lower()
        if var not in EXCLUDED_PUBLIC:
            topics[var] = str(row['NaturalLanguageClause']).strip()
    return topics


def load_all_public_topics() -> Dict[str, str]:
    """Load all public issue topics with no exclusion filter."""
    csv_path = QUESTION_LISTS_DIR / "public_issues.csv"
    df = pd.read_csv(csv_path)
    topics = {}
    for _, row in df.iterrows():
        var = str(row['Variable']).lower()
        nlc = str(row.get('NaturalLanguageClause', '')).strip()
        if nlc and nlc != 'nan':
            topics[var] = nlc
    return topics


def load_all_private_topics() -> Dict[str, str]:
    """Load all private-life topics with no exclusion filter."""
    csv_path = QUESTION_LISTS_DIR / "private_life.csv"
    df = pd.read_csv(csv_path)
    topics = {}
    for _, row in df.iterrows():
        var = str(row['Variable']).lower()
        nlc = str(row.get('NaturalLanguageClause', '')).strip()
        if not nlc or nlc == 'nan':
            nlc = str(row.get('SurveyQuestion', '')).strip()
        if nlc and nlc != 'nan':
            topics[var] = nlc
    return topics


def build_topic_list(topic_set: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (variable, nlc, category) tuples.

    topic_set:
        'public'  — 126 filtered public topics
        'private' — all private topics (no exclusion filter)
        'all'     — all public + all private (no exclusion filters)
    """
    items = []
    if topic_set == 'public':
        for v, d in load_filtered_public_topics().items():
            items.append((v, d, 'public'))
    elif topic_set == 'private':
        for v, d in load_all_private_topics().items():
            items.append((v, d, 'private'))
    elif topic_set == 'all':
        for v, d in load_all_public_topics().items():
            items.append((v, d, 'public'))
        for v, d in load_all_private_topics().items():
            items.append((v, d, 'private'))
    else:
        raise ValueError(f"Unknown topic_set: {topic_set}")
    return items


def load_question_text() -> Dict[str, str]:
    """Load NLC / survey text for survey-style prompts (both public + private)."""
    q_map = {}
    for csv_name in ["public_issues.csv", "private_life.csv"]:
        csv_path = QUESTION_LISTS_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                var = str(row['Variable']).lower()
                nlc = str(row.get('NaturalLanguageClause', '')).strip()
                if nlc and nlc != 'nan':
                    q_map[var] = nlc
                elif pd.notna(row.get('SurveyQuestion', None)):
                    q_map[var] = str(row['SurveyQuestion']).strip()
    return q_map


# =============================================================================
# Prompt generation
# =============================================================================

def generate_prompts(topic_desc: str, question_text: str,
                     politician_names: List[str], model_type: str,
                     condition: str) -> List[str]:
    """Generate prompts for a specific condition and model type."""
    prompts = []
    for name in politician_names:
        if condition == "rhetorical":
            if model_type == "base":
                prompt = f"{name} makes a statement on {topic_desc}:"
            else:
                prompt = f"Generate a statement by {name} on {topic_desc}."
        elif condition == "stance":
            if model_type == "base":
                prompt = f"On {topic_desc}, {name}'s position is"
            else:
                prompt = f"What is {name}'s position on {topic_desc}?"
        elif condition == "survey":
            if model_type == "base":
                prompt = f"In a survey about {question_text}, {name} would respond"
            else:
                prompt = f"If asked in a national survey about {question_text}, how would {name} respond?"
        else:
            raise ValueError(f"Unknown condition: {condition}")
        prompts.append(prompt)
    return prompts


# =============================================================================
# Per-head Mahalanobis computation
# =============================================================================

PCA_DIM = 15

def compute_both_metrics(activations: np.ndarray, labels: np.ndarray) -> dict:
    """
    From (N, L, H, D) activations, compute per-head Mahalanobis grid (L, H)
    then derive:
    - all-layer: mean of full (L, H) grid
    - mid-10%:   mean of grid[mid_start:mid_end, :]
    """
    N, L, H, D = activations.shape

    grid = compute_all_head_metrics_pca(
        activations, labels, group_values=(100, 200),
        n_components=PCA_DIM, centroid_method='mean',
    )  # (L, H)

    mid_start = int(L * 0.45)
    mid_end = max(int(L * 0.55), mid_start + 1)

    return {
        'mahal_all': float(np.mean(grid)),
        'mahal_mid10': float(np.mean(grid[mid_start:mid_end, :])),
        'n_layers': L,
        'mid_layers': f"{mid_start}-{mid_end-1}",
        'n_mid_layers': mid_end - mid_start,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument(
        '--topics', type=str, default='public',
        choices=['public', 'private', 'all'],
        help=(
            "Topic set to run: "
            "'public' = 126 filtered public topics (default); "
            "'private' = all private-life topics incl. excluded; "
            "'all' = all public + all private, no exclusion filters"
        ),
    )
    args = parser.parse_args()

    group = args.group
    topic_set = args.topics
    model_configs = MODEL_GROUPS[group]

    print(f"{'='*80}")
    print(f"EXP 0A/0B EXTENDED — GROUP {group}  topics={topic_set}")
    print(f"Models: {list(model_configs.keys())}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    set_random_seeds(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Experiment tag for output filenames
    exp_tag = EXPERIMENT_NAME if topic_set == 'public' else f"{EXPERIMENT_NAME}_{topic_set}"

    # Load topic list
    topic_list = build_topic_list(topic_set)
    q_text = load_question_text()
    n_pub = sum(1 for _, _, c in topic_list if c == 'public')
    n_prv = sum(1 for _, _, c in topic_list if c == 'private')
    print(f"Topics: {len(topic_list)} (public={n_pub}, private={n_prv})")

    # Load politicians
    df_pol = pd.read_csv(POLITICIAN_CSV)
    df_pol = df_pol[df_pol['party_code'].isin([100, 200])].dropna(subset=['fullname'])
    politician_names = df_pol['fullname'].tolist()
    party_labels = df_pol['party_code'].values
    print(f"Loaded {len(politician_names)} politicians")

    # Load GSS polarization data (public + private combined)
    pol_data = load_polarization_data()
    print(f"Loaded polarization data: {len(pol_data)} variables")

    all_results = []
    n_topics = len(topic_list)

    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({config['type']})")
        print(f"{'='*60}")

        model, tokenizer = load_model(config["path"])
        model_type = config["type"]

        if model_type == "base":
            tokenizer.chat_template = None
            print("  Disabled chat template for base model")

        system_msg = (
            "" if model_type == "base"
            else config.get("system_override", SYSTEM_MSG_POLITICIAN)
        )

        model_start = time.time()

        for cond in CONDITIONS:
            print(f"\n  --- Condition: {cond} ---")
            cond_start = time.time()

            for i, (topic, topic_desc, category) in enumerate(topic_list):
                question = q_text.get(topic, topic_desc)

                prompts = generate_prompts(
                    topic_desc, question, politician_names, model_type, cond
                )

                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=config["batch_size"], max_length=MAX_LENGTH,
                )

                metrics = compute_both_metrics(activations, party_labels)

                pol_row = pol_data[pol_data['variable'] == topic]
                gss_pol = pol_row['polarization'].values[0] if len(pol_row) > 0 else np.nan

                all_results.append({
                    'model': model_name,
                    'model_type': model_type,
                    'family': config['family'],
                    'condition': cond,
                    'category': category,
                    'topic': topic,
                    'mahal_all': metrics['mahal_all'],
                    'mahal_mid10': metrics['mahal_mid10'],
                    'n_layers': metrics['n_layers'],
                    'mid_layers': metrics['mid_layers'],
                    'gss_polarization': gss_pol,
                })

                if (i + 1) % 20 == 0 or i == 0:
                    print(
                        f"    [{i+1}/{n_topics}] {topic:15s} [{category[:3]}] "
                        f"all={metrics['mahal_all']:.3f} "
                        f"mid10={metrics['mahal_mid10']:.3f} "
                        f"gss={'nan' if np.isnan(gss_pol) else f'{gss_pol:.3f}'}"
                    )

                del activations
                gc.collect()
                torch.cuda.empty_cache()

            elapsed = time.time() - cond_start
            print(f"  Condition {cond} done in {elapsed:.0f}s ({elapsed/n_topics:.1f}s/topic)")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - model_start
        print(f"\n  Model {model_name} total: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Save per-model checkpoint
        df_ckpt = pd.DataFrame(all_results)
        ckpt_path = RESULTS_DIR / f"{exp_tag}_g{group}_{model_name}_{timestamp}.pkl"
        df_ckpt.to_pickle(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.name}")

    # ==========================================================================
    # Save final results
    # ==========================================================================
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"{exp_tag}_g{group}_{timestamp}.csv"
    pkl_path = RESULTS_DIR / f"{exp_tag}_g{group}_{timestamp}.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"\nSaved: {csv_path.name}")
    print(f"Rows: {len(df)} | Topics: {df['topic'].nunique()} | "
          f"Models: {df['model'].nunique()} | Conditions: {df['condition'].nunique()}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 0A/0B EXTENDED COMPLETE — GROUP {group} ({topic_set})")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
