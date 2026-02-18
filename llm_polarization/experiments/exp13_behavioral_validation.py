"""
Experiment 13: Behavioral Validation (Generation-Level)

Research Question:
    Does the activation-level partisan signal actually predict model outputs?
    Do instruct models generate more partisan text than base/reasoning models?

Background:
    Experiments 1-12 are all representational: they measure activations, not behavior.
    The ICLR 2025 paper (Kaplan et al.) showed steering is possible but didn't
    quantify whether activation-level partisan distance predicts generation content
    without intervention. This experiment bridges that gap.

Hypotheses:
    H13a: Generated text from instruct models is more partisan (higher partisan
           classifier confidence) than base/reasoning models
    H13b: Topics with higher activation-level Mahalanobis distance produce more
           partisan generated text
    H13c: Activation-level party classification accuracy correlates with
           generation-level party classification accuracy

Method:
    1. For each model x topic x politician subset:
       - Extract activations (as in Exp1)
       - Generate short responses (50 tokens)
    2. Classify generated text as Democrat/Republican using keyword-based scoring
       and a separate LLM classifier (the instruct model itself as zero-shot judge)
    3. Correlate activation-level metrics with generation-level metrics
    4. Compare partisan content across model types

Design:
    - 10 high-polarization topics (subset of exp5)
    - 8 models
    - 50 politicians per party (100 total, for speed)
    - Generate 50 tokens per prompt
    - Two evaluation methods:
      a) Keyword-based partisan scoring (fast, transparent)
      b) Self-classification: ask the model itself to classify the text (when instruct)

Runtime: ~4 hours on H100 (generation is slower than extraction)
"""

import os
import sys
import json
import gc
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, ttest_ind
import torch

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    set_random_seeds,
    load_polarization_data,
    save_checkpoint,
    setup_plot_style,
    save_figure,
    compute_pca_and_distance,
    MODEL_FAMILIES,
    RESULTS_DIR,
    TOPIC_LISTS_DIR,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from config import SYSTEM_MSG_POLITICIAN

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp13"
MAX_LENGTH = 128
GENERATION_LENGTH = 80  # Generate 80 new tokens
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
N_POLITICIANS_PER_PARTY = 50  # Use 50 per party for speed
PCA_DIM = 15

BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}
INSTRUCT_TEMPLATE_KEYS = {
    "public_issues": "default",
    "private_life":  "opinion",
}

# Partisan keyword lists for scoring generated text
DEMOCRAT_KEYWORDS = [
    'equality', 'rights', 'justice', 'progressive', 'affordable',
    'universal', 'inclusion', 'diversity', 'climate', 'renewable',
    'workers', 'union', 'minimum wage', 'healthcare', 'environment',
    'regulation', 'gun control', 'common sense', 'safety net',
    'community', 'systemic', 'reform', 'protect', 'invest',
    'public', 'access', 'opportunity', 'fair', 'equity',
]

REPUBLICAN_KEYWORDS = [
    'freedom', 'liberty', 'constitution', 'individual', 'free market',
    'limited government', 'tax cut', 'deregulation', 'border',
    'security', 'traditional', 'values', 'faith', 'family',
    'second amendment', 'right to bear', 'law and order', 'strong',
    'national defense', 'patriot', 'enterprise', 'business',
    'fiscal', 'responsibility', 'sovereignty', 'american',
    'pro-life', 'religious liberty', 'personal responsibility',
]


# =============================================================================
# Text Generation
# =============================================================================

def generate_text_batch(
    model,
    tokenizer,
    prompts: List[str],
    system_msg: str,
    max_new_tokens: int = 80,
    batch_size: int = 16,
    temperature: float = 0.7,
) -> List[str]:
    """
    Generate text from model given prompts.

    Returns list of generated text strings (only the new tokens, not the prompt).
    """
    model.eval()
    all_generated = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # Format prompts
        formatted = []
        for prompt in batch_prompts:
            if tokenizer.chat_template is not None and system_msg:
                messages = []
                if system_msg:
                    messages.append({"role": "system", "content": system_msg})
                messages.append({"role": "user", "content": prompt})
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                if system_msg:
                    text = f"{system_msg}\n\n{prompt}"
                else:
                    text = prompt
            formatted.append(text)

        # Tokenize
        encodings = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(model.device)

        prompt_length = encodings['input_ids'].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        for j, output in enumerate(outputs):
            new_tokens = output[prompt_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_generated.append(generated_text.strip())

    return all_generated


# =============================================================================
# Partisan Scoring
# =============================================================================

def keyword_partisan_score(text: str) -> Dict[str, float]:
    """
    Score text on partisan keywords.

    Returns dict with:
    - dem_score: count of Democrat keywords
    - rep_score: count of Republican keywords
    - partisan_direction: rep_score - dem_score (positive = more Republican)
    - partisan_magnitude: abs(partisan_direction)
    """
    text_lower = text.lower()

    dem_count = sum(1 for kw in DEMOCRAT_KEYWORDS if kw in text_lower)
    rep_count = sum(1 for kw in REPUBLICAN_KEYWORDS if kw in text_lower)

    total = dem_count + rep_count
    if total == 0:
        return {
            'dem_score': 0,
            'rep_score': 0,
            'partisan_direction': 0.0,
            'partisan_magnitude': 0.0,
            'partisan_fraction': 0.0,
        }

    return {
        'dem_score': dem_count,
        'rep_score': rep_count,
        'partisan_direction': (rep_count - dem_count) / total,
        'partisan_magnitude': abs(rep_count - dem_count) / total,
        'partisan_fraction': total / max(len(text_lower.split()), 1),
    }


def classify_text_partisan(text: str) -> int:
    """
    Simple keyword-based classification: 0=Democrat, 1=Republican.
    Returns -1 if no keywords found.
    """
    score = keyword_partisan_score(text)
    if score['dem_score'] == 0 and score['rep_score'] == 0:
        return -1
    return 1 if score['rep_score'] > score['dem_score'] else 0


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run Experiment 13: Behavioral Validation."""

    print("=" * 80)
    print("EXPERIMENT 13: BEHAVIORAL VALIDATION (GENERATION-LEVEL)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    set_random_seeds(42)

    # =========================================================================
    # Load data
    # =========================================================================

    print("Loading politicians...")
    df_politicians = pd.read_csv(POLITICIAN_CSV)
    df_dr = df_politicians[df_politicians['party_code'].isin([100, 200])].copy()

    # Sample N per party for speed
    rng = np.random.RandomState(42)
    dem_idx = df_dr[df_dr['party_code'] == 100].index.tolist()
    rep_idx = df_dr[df_dr['party_code'] == 200].index.tolist()
    sampled_dem = rng.choice(dem_idx, size=min(N_POLITICIANS_PER_PARTY, len(dem_idx)), replace=False)
    sampled_rep = rng.choice(rep_idx, size=min(N_POLITICIANS_PER_PARTY, len(rep_idx)), replace=False)
    sampled_idx = np.concatenate([sampled_dem, sampled_rep])
    df_sample = df_dr.loc[sampled_idx].reset_index(drop=True)

    politician_names = df_sample['fullname'].tolist()
    party_labels = (df_sample['party_code'].values == 200).astype(int)
    n_dem = int(np.sum(party_labels == 0))
    n_rep = int(np.sum(party_labels == 1))
    print(f"Sampled {len(politician_names)} politicians (D={n_dem}, R={n_rep})")

    # Load topics - 10 high-polarization topics
    print("\nLoading topics...")
    topic_list_path = TOPIC_LISTS_DIR / "exp13_behavioral_topics.json"
    if topic_list_path.exists():
        with open(topic_list_path) as f:
            topics = json.load(f)
    else:
        # Use a mix of polarized topics
        topics = {
            "gunlaw": {"description": "gun control laws", "category": "public_issues"},
            "abany": {"description": "abortion for any reason", "category": "public_issues"},
            "cappun": {"description": "the death penalty", "category": "public_issues"},
            "eqwlth": {"description": "government reducing income inequality", "category": "public_issues"},
            "natenvir": {"description": "government spending on the environment", "category": "public_issues"},
            "homosex": {"description": "homosexual relations", "category": "private_life"},
            "prayer": {"description": "prayer in public schools", "category": "private_life"},
            "grass": {"description": "marijuana legalization", "category": "private_life"},
            "polviews": {"description": "political ideology and party affiliation", "category": "public_issues"},
            "immig": {"description": "immigration policy", "category": "public_issues"},
        }
        with open(topic_list_path, 'w') as f:
            json.dump(topics, f, indent=2)
        print(f"  Created and saved topic list: {topic_list_path}")

    print(f"Using {len(topics)} topics")

    # =========================================================================
    # Run all models
    # =========================================================================

    all_results = []

    for family_name, family_config in MODEL_FAMILIES.items():
        print(f"\n{'='*80}")
        print(f"FAMILY: {family_name}")
        print(f"{'='*80}")

        for variant_name, variant_cfg in family_config.items():
            model_path = variant_cfg['path']
            model_type = variant_cfg['type']
            batch_size = variant_cfg['batch_size']
            model_name = f"{family_name}_{variant_name}"

            # Use smaller batch for generation (more memory intensive)
            gen_batch_size = max(8, batch_size // 4)

            print(f"\n--- Running {model_name} ({model_type}) ---")

            # Load model
            model, tokenizer = load_model(model_path)
            if model_type == "base":
                tokenizer.chat_template = None

            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            for topic_idx, (topic_name, topic_info) in enumerate(topics.items(), 1):
                topic_desc = topic_info['description']
                category = topic_info['category']

                print(f"\n  [{topic_idx}/{len(topics)}] {topic_name} ({category})")

                # Generate prompts
                if model_type == "base":
                    template = BASE_TEMPLATES[category]
                    prompts = [
                        template.format(name=name, topic=topic_desc)
                        for name in politician_names
                    ]
                    system_msg = ""
                else:
                    key = INSTRUCT_TEMPLATE_KEYS[category]
                    template_str = POLITICIAN_TEMPLATES[key]
                    prompts = [
                        template_str.format(name=name, topic=topic_desc)
                        for name in politician_names
                    ]
                    system_msg = SYSTEM_MSG_POLITICIAN

                # --- Step 1: Extract activations ---
                print(f"    Extracting activations...")
                activations = extract_heads_batched(
                    model, tokenizer, prompts, system_msg,
                    batch_size=batch_size, max_length=MAX_LENGTH
                )
                if hasattr(activations, 'cpu'):
                    activations_np = activations.cpu().numpy()
                elif isinstance(activations, np.ndarray):
                    activations_np = activations
                else:
                    activations_np = np.array(activations)

                # Compute Mahalanobis distance
                try:
                    pca_result = compute_pca_and_distance(
                        activations_np, party_labels, pca_dim=PCA_DIM
                    )
                    mahal_dist = pca_result['mahalanobis_dist']
                    var_pc1 = pca_result['variance_explained']
                except Exception:
                    mahal_dist = 0.0
                    var_pc1 = 0.0

                del activations, activations_np
                gc.collect()
                torch.cuda.empty_cache()

                # --- Step 2: Generate text ---
                print(f"    Generating text ({GENERATION_LENGTH} tokens)...")
                generated_texts = generate_text_batch(
                    model, tokenizer, prompts, system_msg,
                    max_new_tokens=GENERATION_LENGTH,
                    batch_size=gen_batch_size,
                    temperature=0.7,
                )

                # --- Step 3: Score generated text ---
                print(f"    Scoring partisan content...")
                dem_directions = []
                rep_directions = []
                dem_correct = 0
                rep_correct = 0
                dem_total = 0
                rep_total = 0

                for p_idx, (text, party) in enumerate(zip(generated_texts, party_labels)):
                    score = keyword_partisan_score(text)
                    pred_class = classify_text_partisan(text)

                    if party == 0:  # Democrat
                        dem_directions.append(score['partisan_direction'])
                        if pred_class == 0:
                            dem_correct += 1
                        if pred_class >= 0:
                            dem_total += 1
                    else:  # Republican
                        rep_directions.append(score['partisan_direction'])
                        if pred_class == 1:
                            rep_correct += 1
                        if pred_class >= 0:
                            rep_total += 1

                # Compute generation-level metrics
                dem_mean_dir = np.mean(dem_directions) if dem_directions else 0.0
                rep_mean_dir = np.mean(rep_directions) if rep_directions else 0.0
                gen_separation = rep_mean_dir - dem_mean_dir  # Should be positive
                gen_accuracy = 0.0
                if dem_total + rep_total > 0:
                    gen_accuracy = (dem_correct + rep_correct) / (dem_total + rep_total)

                # Count empty/unscored
                n_unscored = sum(1 for t in generated_texts if classify_text_partisan(t) == -1)

                print(f"    Activation Mahalanobis: {mahal_dist:.3f}")
                print(f"    Generation separation: {gen_separation:.3f}")
                print(f"    Generation accuracy: {gen_accuracy:.3f}")
                print(f"    Unscored texts: {n_unscored}/{len(generated_texts)}")

                all_results.append({
                    'family': family_name,
                    'variant': variant_name,
                    'model_name': model_name,
                    'model_type': model_type,
                    'topic_name': topic_name,
                    'category': category,
                    'activation_mahal': mahal_dist,
                    'activation_var_pc1': var_pc1,
                    'gen_dem_direction': dem_mean_dir,
                    'gen_rep_direction': rep_mean_dir,
                    'gen_separation': gen_separation,
                    'gen_accuracy': gen_accuracy,
                    'n_unscored': n_unscored,
                    'n_total': len(generated_texts),
                })

                # Save checkpoint
                save_checkpoint(
                    all_results,
                    EXPERIMENT_NAME, f"{model_name}_checkpoint"
                )

                del generated_texts
                gc.collect()
                torch.cuda.empty_cache()

            # Cleanup model
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n  Unloaded {model_name}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_behavioral_{timestamp}.csv"
    df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    # --- Generation separation by model type ---
    print("\n--- Generation Partisan Separation by Model Type ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df[df['model_type'] == mt]
        if len(sub) == 0:
            continue
        print(f"  {mt}:")
        print(f"    Gen separation: {sub['gen_separation'].mean():.4f} ± {sub['gen_separation'].std():.4f}")
        print(f"    Gen accuracy: {sub['gen_accuracy'].mean():.4f} ± {sub['gen_accuracy'].std():.4f}")
        print(f"    Activation Mahal: {sub['activation_mahal'].mean():.4f}")
        print(f"    Unscored rate: {sub['n_unscored'].sum() / sub['n_total'].sum():.3f}")

    # --- Correlation: activation distance vs generation separation ---
    print("\n--- Correlation: Activation Mahalanobis vs Generation Separation ---")
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df[df['model_type'] == mt]
        if len(sub) < 3:
            continue
        rho, p = spearmanr(sub['activation_mahal'], sub['gen_separation'])
        r, p_r = pearsonr(sub['activation_mahal'], sub['gen_separation'])
        print(f"  {mt}: Spearman rho={rho:.4f} (p={p:.4f}), Pearson r={r:.4f} (p={p_r:.4f})")

    # Overall correlation
    rho, p = spearmanr(df['activation_mahal'], df['gen_separation'])
    print(f"  Overall: Spearman rho={rho:.4f} (p={p:.4f})")

    # --- Statistical tests ---
    print("\n--- Statistical Tests (gen_separation) ---")
    for mt1, mt2 in [('base', 'instruct'), ('base', 'reasoning'), ('instruct', 'reasoning')]:
        g1 = df[df['model_type'] == mt1]['gen_separation']
        g2 = df[df['model_type'] == mt2]['gen_separation']
        if len(g1) > 0 and len(g2) > 0:
            t, p = ttest_ind(g1, g2)
            print(f"  {mt1} vs {mt2}: t={t:.3f}, p={p:.4e}")

    # --- Plots ---
    setup_plot_style()

    # Scatter: activation Mahalanobis vs generation separation
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'base': '#2196F3', 'instruct': '#FF5722', 'reasoning': '#4CAF50'}
    for mt in ['base', 'instruct', 'reasoning']:
        sub = df[df['model_type'] == mt]
        ax.scatter(sub['activation_mahal'], sub['gen_separation'],
                   c=colors.get(mt, 'gray'), label=mt, alpha=0.7, s=40)
    ax.set_xlabel('Activation-Level Mahalanobis Distance')
    ax.set_ylabel('Generation-Level Partisan Separation')
    ax.set_title('Behavioral Validation: Activations vs Generation')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    save_figure(fig, EXPERIMENT_NAME, f"scatter_{timestamp}")

    # Bar chart: generation separation by model type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: generation separation
    ax = axes[0]
    plot_data = df.groupby('model_type')['gen_separation'].agg(['mean', 'std']).reindex(['base', 'instruct', 'reasoning'])
    ax.bar(plot_data.index, plot_data['mean'], yerr=plot_data['std'],
           color=[colors.get(x, 'gray') for x in plot_data.index],
           capsize=5, alpha=0.8)
    ax.set_ylabel('Partisan Separation (R - D keyword direction)')
    ax.set_title('Generation Partisan Separation')

    # Right: generation accuracy
    ax = axes[1]
    plot_data = df.groupby('model_type')['gen_accuracy'].agg(['mean', 'std']).reindex(['base', 'instruct', 'reasoning'])
    ax.bar(plot_data.index, plot_data['mean'], yerr=plot_data['std'],
           color=[colors.get(x, 'gray') for x in plot_data.index],
           capsize=5, alpha=0.8)
    ax.set_ylabel('Generation-Level Party Classification Accuracy')
    ax.set_title('Keyword-Based Party Classification')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
    ax.legend()

    fig.tight_layout()
    save_figure(fig, EXPERIMENT_NAME, f"bars_{timestamp}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 13 COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_experiment()
