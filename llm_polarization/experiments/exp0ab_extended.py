"""
Experiment 0A/0B Extended: Prompt Framing + Layer Selection on Full Public Issues

0A: Does prompt type (rhetorical/stance/survey) affect Mahalanobis-GSS correlation?
0B: Does restricting to middle 10% of layers improve correlation?

Uses 126 filtered public issues (134 minus 8 excluded) across 10 model variants
in 4 families: Gemma-2-9b, Llama-3.1-8B, Qwen3-4B, SmolLM3-3B.

Usage:
    python exp0ab_extended.py --group 1   # gemma-2-9b (base + instruct)
    python exp0ab_extended.py --group 2   # Llama-3.1-8B (base + instruct)
    python exp0ab_extended.py --group 3   # Qwen3-4B (base + instruct + reasoning)
    python exp0ab_extended.py --group 4   # SmolLM3-3B (base + instruct + reasoning)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import gc
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import set_random_seeds, RESULTS_DIR, load_polarization_data

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians
from config import SYSTEM_MSG_POLITICIAN
from run_gss_pca import compute_all_head_metrics_pca

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp0ab_ext"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
QUESTION_LISTS_DIR = Path("/project/jevans/maxzhuyt/gss_polarization/question_lists")
MODELS_DIR = "/project/jevans/maxzhuyt/models"

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
# Topic + question text loading
# =============================================================================

def load_filtered_public_topics() -> Dict[str, str]:
    """Load 126 filtered public issue topics (134 - 8 excluded)."""
    csv_path = QUESTION_LISTS_DIR / "public_issues.csv"
    df = pd.read_csv(csv_path)
    topics = {}
    for _, row in df.iterrows():
        var = row['Variable'].lower()
        if var not in EXCLUDED_PUBLIC:
            topics[var] = str(row['NaturalLanguageClause']).strip()
    return topics


def load_question_text() -> Dict[str, str]:
    """Load GSS question text for survey-style prompts."""
    q_map = {}
    for csv_name in ["public_issues.csv", "private_life.csv"]:
        csv_path = QUESTION_LISTS_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                var = row['Variable'].lower()
                if pd.notna(row.get('NaturalLanguageClause', None)) and str(row['NaturalLanguageClause']).strip():
                    q_map[var] = str(row['NaturalLanguageClause']).strip()
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
# Per-head Mahalanobis computation (matches run_model_comparison.py baseline)
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
    args = parser.parse_args()

    group = args.group
    model_configs = MODEL_GROUPS[group]

    print(f"{'='*80}")
    print(f"EXP 0A/0B EXTENDED — GROUP {group}")
    print(f"Models: {list(model_configs.keys())}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    set_random_seeds(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    topics = load_filtered_public_topics()
    q_text = load_question_text()
    print(f"Loaded {len(topics)} filtered public topics")

    df_pol = pd.read_csv(POLITICIAN_CSV)
    df_pol = df_pol[df_pol['party_code'].isin([100, 200])].dropna(subset=['fullname'])
    politician_names = df_pol['fullname'].tolist()
    party_labels = df_pol['party_code'].values
    print(f"Loaded {len(politician_names)} politicians")

    pol_data = load_polarization_data()
    print(f"Loaded polarization data: {len(pol_data)} variables")

    all_results = []
    topic_list = list(topics.items())
    n_topics = len(topic_list)

    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({config['type']})")
        print(f"{'='*60}")

        model, tokenizer = load_model(config["path"])
        model_type = config["type"]

        # Disable chat template for base models
        if model_type == "base":
            tokenizer.chat_template = None
            print("  Disabled chat template for base model")

        if model_type == "base":
            system_msg = ""
        else:
            system_msg = config.get("system_override", SYSTEM_MSG_POLITICIAN)

        model_start = time.time()

        for cond in CONDITIONS:
            print(f"\n  --- Condition: {cond} ---")
            cond_start = time.time()

            for i, (topic, topic_desc) in enumerate(topic_list):
                question = q_text.get(topic, topic_desc)

                prompts = generate_prompts(
                    topic_desc, question, politician_names, model_type, cond
                )

                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=config["batch_size"], max_length=MAX_LENGTH,
                )

                metrics = compute_both_metrics(activations, party_labels)

                # GSS polarization
                pol_row = pol_data[pol_data['variable'] == topic]
                gss_pol = pol_row['polarization'].values[0] if len(pol_row) > 0 else np.nan

                all_results.append({
                    'model': model_name,
                    'model_type': model_type,
                    'family': config['family'],
                    'condition': cond,
                    'topic': topic,
                    'mahal_all': metrics['mahal_all'],
                    'mahal_mid10': metrics['mahal_mid10'],
                    'n_layers': metrics['n_layers'],
                    'mid_layers': metrics['mid_layers'],
                    'gss_polarization': gss_pol,
                })

                if (i + 1) % 20 == 0 or i == 0:
                    print(f"    [{i+1}/{n_topics}] {topic:12s} all={metrics['mahal_all']:.3f} mid10={metrics['mahal_mid10']:.3f} gss={gss_pol:.3f}")

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
        ckpt_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{group}_{model_name}_{timestamp}.pkl"
        df_ckpt.to_pickle(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.name}")

    # ==========================================================================
    # Save final results
    # ==========================================================================
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{group}_{timestamp}.csv"
    pkl_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_g{group}_{timestamp}.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"\nSaved: {csv_path.name}")

    # ==========================================================================
    # Analysis
    # ==========================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS — EXP 0A: PROMPT FRAMING (all-layer Mahalanobis)")
    print(f"{'='*80}")

    print(f"\n{'Model':30s} {'Cond':12s} {'r':>7s} {'p':>8s} {'rho':>7s} {'p_rho':>8s} {'n':>4s}")
    print("-" * 80)

    for (model, cond), grp in df.groupby(['model', 'condition']):
        valid = grp.dropna(subset=['gss_polarization'])
        if len(valid) >= 5:
            r, p = pearsonr(valid['mahal_all'], valid['gss_polarization'])
            rho, p_rho = spearmanr(valid['mahal_all'], valid['gss_polarization'])
            print(f"{model:30s} {cond:12s} {r:7.3f} {p:8.4f} {rho:7.3f} {p_rho:8.4f} {len(valid):4d}")

    # Average r by condition
    print(f"\n--- Average r by Condition ---")
    for cond in CONDITIONS:
        rs = []
        for model in df['model'].unique():
            sub = df[(df['model'] == model) & (df['condition'] == cond)].dropna(subset=['gss_polarization'])
            if len(sub) >= 5:
                r, _ = pearsonr(sub['mahal_all'], sub['gss_polarization'])
                rs.append(r)
        if rs:
            print(f"  {cond:12s}: mean_r={np.mean(rs):.3f}, median_r={np.median(rs):.3f} (n_models={len(rs)})")

    print(f"\n{'='*80}")
    print("ANALYSIS — EXP 0B: MIDDLE 10% vs ALL LAYERS")
    print(f"{'='*80}")

    # Use rhetorical condition as the default for 0b comparison
    df_rhet = df[df['condition'] == 'rhetorical']

    print(f"\n{'Model':30s} {'Metric':10s} {'r':>7s} {'p':>8s} {'rho':>7s} {'p_rho':>8s}")
    print("-" * 80)

    for model in df_rhet['model'].unique():
        sub = df_rhet[df_rhet['model'] == model].dropna(subset=['gss_polarization'])
        if len(sub) >= 5:
            for metric, label in [('mahal_all', 'all-layer'), ('mahal_mid10', 'mid-10%')]:
                r, p = pearsonr(sub[metric], sub['gss_polarization'])
                rho, p_rho = spearmanr(sub[metric], sub['gss_polarization'])
                print(f"{model:30s} {label:10s} {r:7.3f} {p:8.4f} {rho:7.3f} {p_rho:8.4f}")

    # Also show 0b for all conditions (bonus)
    print(f"\n--- Mid-10% r by Model × Condition ---")
    print(f"{'Model':30s} {'Cond':12s} {'r_all':>7s} {'r_mid10':>9s} {'delta':>7s}")
    print("-" * 70)

    for (model, cond), grp in df.groupby(['model', 'condition']):
        valid = grp.dropna(subset=['gss_polarization'])
        if len(valid) >= 5:
            r_all, _ = pearsonr(valid['mahal_all'], valid['gss_polarization'])
            r_mid, _ = pearsonr(valid['mahal_mid10'], valid['gss_polarization'])
            delta = r_mid - r_all
            marker = " +" if delta > 0 else ""
            print(f"{model:30s} {cond:12s} {r_all:7.3f} {r_mid:9.3f} {marker}{delta:6.3f}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 0A/0B EXTENDED COMPLETE — GROUP {group}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
