"""
Experiment 0A: Prompt Framing — Rhetoric vs Opinion vs Survey

Research Question:
    Does the way we prompt the model (rhetorical statement vs opinion stance vs
    survey-style response) affect how well LLM activation polarization aligns
    with GSS survey polarization?

Hypothesis:
    Survey-aligned prompts will better predict GSS polarization because they
    elicit opinion-like representations rather than rhetorical framing.

Method:
    For 20 strategically selected topics (5 over-polarized, 5 under-polarized,
    5 well-aligned high-pol, 5 well-aligned low-pol):
    - Condition 1 (RHETORICAL): "Generate a statement by {name} on {topic}"
    - Condition 2 (STANCE): "What is {name}'s position on {topic}?"
    - Condition 3 (SURVEY): "If asked in a national survey: '{GSS question}',
      how would {name} respond?"
    - Extract activations, compute Mahalanobis (PCA-15), correlate with GSS.

Models: Qwen3-4B (base/instruct/reasoning) + SmolLM3-3B (base/instruct/reasoning)
Runtime: ~20 min on A100
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr

# Imports
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    set_random_seeds, save_checkpoint, RESULTS_DIR,
    load_polarization_data,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians
from config import SYSTEM_MSG_POLITICIAN, TOPICS_GSS

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp0a"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
QUESTION_LISTS_DIR = Path("/project/jevans/maxzhuyt/gss_polarization/question_lists")

# Only Qwen3-4B and SmolLM3-3B
MODELS_DIR = "/project/jevans/maxzhuyt/models"
MODEL_CONFIGS = {
    "Qwen3-4B_base": {"path": f"{MODELS_DIR}/Qwen3-4B-Base", "type": "base", "batch_size": 180, "family": "Qwen3-4B"},
    "Qwen3-4B_instruct": {"path": f"{MODELS_DIR}/Qwen3-4B-Instruct-2507", "type": "instruct", "batch_size": 180, "family": "Qwen3-4B"},
    "Qwen3-4B_reasoning": {"path": f"{MODELS_DIR}/Qwen3-4B-Thinking-2507", "type": "reasoning", "batch_size": 180, "family": "Qwen3-4B"},
    "SmolLM3-3B_base": {"path": f"{MODELS_DIR}/SmolLM3-3B-Base", "type": "base", "batch_size": 200, "family": "SmolLM3-3B"},
    "SmolLM3-3B_instruct": {"path": f"{MODELS_DIR}/SmolLM3-3B", "type": "instruct", "batch_size": 200, "family": "SmolLM3-3B",
                             "system_override": "/no_think"},
    "SmolLM3-3B_reasoning": {"path": f"{MODELS_DIR}/SmolLM3-3B", "type": "reasoning", "batch_size": 200, "family": "SmolLM3-3B",
                              "system_override": "/think"},
}

# 20 strategically selected topics with mismatch category
SELECTED_TOPICS = {
    # OVER-POLARIZED: LLM rhetoric >> GSS opinion (residual > +0.05)
    "colhomo": ("public_issues", "over"),     # allow gay teacher
    "spkhomo": ("public_issues", "over"),     # allow gay person to speak
    "oprelig": ("private_life", "over"),      # importance of religion
    "impgrn":  ("public_issues", "over"),     # importance of environment
    "polviews": ("public_issues", "over"),    # liberal-conservative
    # UNDER-POLARIZED: LLM rhetoric << GSS opinion (residual < -0.05)
    "savesoul": ("private_life", "under"),    # try to save others' souls
    "pray":     ("private_life", "under"),    # prayer frequency
    "polescap": ("public_issues", "under"),   # police strike escaper
    "conbus":   ("public_issues", "under"),   # confidence in business
    "conlegis": ("public_issues", "under"),   # confidence in congress
    # WELL-ALIGNED HIGH POLARIZATION
    "eqwlth":   ("public_issues", "aligned_high"),  # govt reduce income diff
    "cappun":   ("public_issues", "aligned_high"),   # death penalty
    "grnexagg": ("public_issues", "aligned_high"),   # env threats exaggerated
    "helpblk":  ("public_issues", "aligned_high"),   # govt aid Black people
    "gunlaw":   ("public_issues", "aligned_high"),   # gun permits
    # WELL-ALIGNED LOW POLARIZATION
    "natspac":  ("public_issues", "aligned_low"),    # spending on space
    "natroad":  ("public_issues", "aligned_low"),    # spending on highways
    "courts":   ("public_issues", "aligned_low"),    # courts harsh/lenient
    "natsoc":   ("public_issues", "aligned_low"),    # spending on social security
    "natsci":   ("public_issues", "aligned_low"),    # spending on science
}

# =============================================================================
# Load GSS question text
# =============================================================================

def load_question_text() -> Dict[str, str]:
    """Load exact GSS survey question text for each variable."""
    q_map = {}
    for csv_name in ["public_issues.csv", "private_life.csv"]:
        csv_path = QUESTION_LISTS_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                var = row['Variable'].lower()
                # Prefer NaturalLanguageClause (cleaner), fall back to SurveyQuestion
                if pd.notna(row.get('NaturalLanguageClause', None)) and str(row['NaturalLanguageClause']).strip():
                    q_map[var] = str(row['NaturalLanguageClause']).strip()
                elif pd.notna(row.get('SurveyQuestion', None)):
                    q_map[var] = str(row['SurveyQuestion']).strip()
    return q_map


# =============================================================================
# Prompt Generation
# =============================================================================

def generate_condition_prompts(
    topic_name: str,
    topic_desc: str,
    question_text: str,
    politician_names: List[str],
    model_type: str,
    condition: str,
) -> List[str]:
    """Generate prompts for a specific condition."""
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
# Mahalanobis computation
# =============================================================================

def compute_mahalanobis_pca(activations: np.ndarray, labels: np.ndarray, n_components: int = 15) -> float:
    """Compute Mahalanobis distance between D and R centroids in PCA space."""
    N, L, H, D = activations.shape
    # Flatten to (N, L*H*D) then mean across layers → (N, H*D)
    flat = activations.mean(axis=1).reshape(N, -1)  # (N, H*D)

    # PCA reduce
    n_comp = min(n_components, flat.shape[0] - 1, flat.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(flat)

    dem_mask = labels == 100
    rep_mask = labels == 200

    if dem_mask.sum() < 2 or rep_mask.sum() < 2:
        return 0.0

    centroid_D = reduced[dem_mask].mean(axis=0)
    centroid_R = reduced[rep_mask].mean(axis=0)

    # Pooled covariance
    cov_D = np.cov(reduced[dem_mask].T)
    cov_R = np.cov(reduced[rep_mask].T)
    n_D, n_R = dem_mask.sum(), rep_mask.sum()
    pooled_cov = ((n_D - 1) * cov_D + (n_R - 1) * cov_R) / (n_D + n_R - 2)

    try:
        cov_inv = np.linalg.pinv(pooled_cov)
        dist = mahalanobis(centroid_D, centroid_R, cov_inv)
    except Exception:
        dist = 0.0

    return dist


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"{'='*80}")
    print(f"EXPERIMENT 0A: PROMPT FRAMING — RHETORIC vs OPINION vs SURVEY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    set_random_seeds(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load question text
    q_text = load_question_text()
    print(f"Loaded question text for {len(q_text)} variables")

    # Load politicians
    df_pol = load_politicians(POLITICIAN_CSV)
    politician_names = df_pol['bioname'].tolist()
    party_labels = df_pol['party_code'].values
    print(f"Loaded {len(politician_names)} politicians")

    # Load polarization data
    pol_data = load_polarization_data()
    print(f"Loaded polarization data: {len(pol_data)} variables")

    # Topic descriptions
    topic_descs = {k: TOPICS_GSS.get(k, k) for k in SELECTED_TOPICS}

    CONDITIONS = ["rhetorical", "stance", "survey"]
    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(config["path"])
        model_type = config["type"]

        for cond in CONDITIONS:
            print(f"\n  Condition: {cond}")

            for i, (topic, (category, mismatch_type)) in enumerate(SELECTED_TOPICS.items()):
                topic_desc = topic_descs.get(topic, topic)
                question = q_text.get(topic, topic_desc)

                prompts = generate_condition_prompts(
                    topic, topic_desc, question,
                    politician_names, model_type, cond
                )

                if model_type == "base":
                    system_msg = ""
                else:
                    system_msg = config.get("system_override", SYSTEM_MSG_POLITICIAN)

                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=config["batch_size"], max_length=MAX_LENGTH
                )

                mahal = compute_mahalanobis_pca(activations, party_labels, n_components=15)

                # Get GSS polarization
                pol_row = pol_data[pol_data['variable'] == topic]
                gss_pol = pol_row['polarization'].values[0] if len(pol_row) > 0 else np.nan

                all_results.append({
                    'model': model_name,
                    'model_type': model_type,
                    'family': config['family'],
                    'condition': cond,
                    'topic': topic,
                    'category': category,
                    'mismatch_type': mismatch_type,
                    'mahalanobis': mahal,
                    'gss_polarization': gss_pol,
                })

                print(f"    [{i+1}/{len(SELECTED_TOPICS)}] {topic:12s} mahal={mahal:.3f} gss={gss_pol:.3f}")

                del activations
                gc.collect()
                torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Save checkpoint
        df_res = pd.DataFrame(all_results)
        save_checkpoint(df_res, f"{EXPERIMENT_NAME}_{model_name}", timestamp)
        print(f"  Saved checkpoint after {model_name}")

    # =============================================================================
    # Analysis
    # =============================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_results_{timestamp}.csv", index=False)

    # Compute correlations per model × condition
    print("\n--- Pearson r (Mahalanobis vs GSS) by Model × Condition ---")
    print(f"{'Model':35s} {'Condition':12s} {'r':>7s} {'p':>8s} {'n':>4s}")
    print("-" * 70)

    corr_rows = []
    for (model, cond), grp in df_res.groupby(['model', 'condition']):
        valid = grp.dropna(subset=['gss_polarization'])
        if len(valid) >= 5:
            r, p = pearsonr(valid['mahalanobis'], valid['gss_polarization'])
            print(f"{model:35s} {cond:12s} {r:7.3f} {p:8.4f} {len(valid):4d}")
            corr_rows.append({'model': model, 'condition': cond, 'pearson_r': r, 'p_value': p, 'n': len(valid)})

    # Summary: average r by condition
    df_corr = pd.DataFrame(corr_rows)
    print(f"\n--- Average Pearson r by Condition (across models) ---")
    for cond in CONDITIONS:
        sub = df_corr[df_corr['condition'] == cond]
        if len(sub) > 0:
            print(f"  {cond:12s}: mean_r = {sub['pearson_r'].mean():.3f}, median_r = {sub['pearson_r'].median():.3f}")

    # Summary: average r by condition × model_type
    print(f"\n--- Average Pearson r by Condition × Model Type ---")
    for cond in CONDITIONS:
        for mtype in ['base', 'instruct', 'reasoning']:
            sub = df_corr[(df_corr['condition'] == cond)]
            sub_models = df_res[df_res['condition'] == cond]['model'].unique()
            type_models = [m for m in sub_models if mtype in m.lower() or (mtype == 'reasoning' and 'reasoning' in MODEL_CONFIGS.get(m, {}).get('type', ''))]
            type_sub = sub[sub['model'].isin(type_models)]
            if len(type_sub) > 0:
                print(f"  {cond:12s} × {mtype:10s}: mean_r = {type_sub['pearson_r'].mean():.3f}")

    # Per-mismatch-type analysis
    print(f"\n--- Mean Mahalanobis by Mismatch Type × Condition ---")
    for mtype in ['over', 'under', 'aligned_high', 'aligned_low']:
        print(f"\n  {mtype}:")
        for cond in CONDITIONS:
            sub = df_res[(df_res['mismatch_type'] == mtype) & (df_res['condition'] == cond)]
            print(f"    {cond:12s}: mean_mahal = {sub['mahalanobis'].mean():.3f} (std={sub['mahalanobis'].std():.3f})")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 0A COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
