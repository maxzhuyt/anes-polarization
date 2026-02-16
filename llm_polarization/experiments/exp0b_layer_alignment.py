"""
Experiment 0B: Per-Layer Alignment Profile

Research Question:
    Which layers of the model best predict GSS survey polarization?
    Is there a separation between "rhetoric layers" (early?) and "opinion layers" (late?)?

Hypothesis:
    If LLMs encode rhetorical framing in different layers than genuine opinion
    content, then the layer-GSS correlation curve should reveal which layers
    capture "opinion-like" information vs "rhetoric-like" information.

Method:
    For 20 selected topics:
    - Extract full (N, L, H, D) activations
    - For EACH layer l: reshape heads to (N, H*D), PCA-reduce, compute Mahalanobis
    - Correlate per-layer Mahalanobis with GSS polarization
    - Find: optimal layer(s), layer profile differences for high/low-mismatch topics

Models: Qwen3-4B (3) + SmolLM3-3B (3)
Runtime: ~15 min on A100
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    set_random_seeds, save_checkpoint, RESULTS_DIR,
    load_polarization_data,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts
from config import SYSTEM_MSG_POLITICIAN, TOPICS_GSS

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp0b"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

MODELS_DIR = "/project/jevans/maxzhuyt/models"
MODEL_CONFIGS = {
    "Qwen3-4B_base": {"path": f"{MODELS_DIR}/Qwen3-4B-Base", "type": "base", "batch_size": 180, "family": "Qwen3-4B"},
    "Qwen3-4B_instruct": {"path": f"{MODELS_DIR}/Qwen3-4B-Instruct-2507", "type": "instruct", "batch_size": 180, "family": "Qwen3-4B"},
    "Qwen3-4B_reasoning": {"path": f"{MODELS_DIR}/Qwen3-4B-Thinking-2507", "type": "reasoning", "batch_size": 180, "family": "Qwen3-4B"},
    "SmolLM3-3B_base": {"path": f"{MODELS_DIR}/SmolLM3-3B", "type": "base", "batch_size": 200, "family": "SmolLM3-3B"},
    "SmolLM3-3B_instruct": {"path": f"{MODELS_DIR}/SmolLM3-3B-Instruct", "type": "instruct", "batch_size": 200, "family": "SmolLM3-3B"},
    "SmolLM3-3B_reasoning": {"path": f"{MODELS_DIR}/SmolLM3-3B-Instruct", "type": "reasoning", "batch_size": 200, "family": "SmolLM3-3B"},
}

# Same 20 topics as exp0a
SELECTED_TOPICS = {
    "colhomo": ("public_issues", "over"),
    "spkhomo": ("public_issues", "over"),
    "oprelig": ("private_life", "over"),
    "impgrn":  ("public_issues", "over"),
    "polviews": ("public_issues", "over"),
    "savesoul": ("private_life", "under"),
    "pray":     ("private_life", "under"),
    "polescap": ("public_issues", "under"),
    "conbus":   ("public_issues", "under"),
    "conlegis": ("public_issues", "under"),
    "eqwlth":   ("public_issues", "aligned_high"),
    "cappun":   ("public_issues", "aligned_high"),
    "grnexagg": ("public_issues", "aligned_high"),
    "helpblk":  ("public_issues", "aligned_high"),
    "gunlaw":   ("public_issues", "aligned_high"),
    "natspac":  ("public_issues", "aligned_low"),
    "natroad":  ("public_issues", "aligned_low"),
    "courts":   ("public_issues", "aligned_low"),
    "natsoc":   ("public_issues", "aligned_low"),
    "natsci":   ("public_issues", "aligned_low"),
}

BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}

# =============================================================================
# Per-layer Mahalanobis
# =============================================================================

def compute_per_layer_mahalanobis(activations: np.ndarray, labels: np.ndarray, n_components: int = 15) -> List[float]:
    """
    Compute Mahalanobis distance at each layer separately.

    activations: (N, L, H, D)
    Returns: list of L Mahalanobis distances
    """
    N, L, H, D = activations.shape
    layer_distances = []

    dem_mask = labels == 100
    rep_mask = labels == 200

    for l in range(L):
        layer_act = activations[:, l, :, :].reshape(N, H * D)  # (N, H*D)

        n_comp = min(n_components, N - 1, layer_act.shape[1])
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(layer_act)

        if dem_mask.sum() < 2 or rep_mask.sum() < 2:
            layer_distances.append(0.0)
            continue

        centroid_D = reduced[dem_mask].mean(axis=0)
        centroid_R = reduced[rep_mask].mean(axis=0)

        cov_D = np.cov(reduced[dem_mask].T)
        cov_R = np.cov(reduced[rep_mask].T)
        n_D, n_R = dem_mask.sum(), rep_mask.sum()
        pooled_cov = ((n_D - 1) * cov_D + (n_R - 1) * cov_R) / (n_D + n_R - 2)

        try:
            cov_inv = np.linalg.pinv(pooled_cov)
            dist = mahalanobis(centroid_D, centroid_R, cov_inv)
        except Exception:
            dist = 0.0

        layer_distances.append(dist)

    return layer_distances


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"{'='*80}")
    print(f"EXPERIMENT 0B: PER-LAYER ALIGNMENT PROFILE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    set_random_seeds(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load
    df_pol = load_politicians(POLITICIAN_CSV)
    politician_names = df_pol['bioname'].tolist()
    party_labels = df_pol['party_code'].values
    pol_data = load_polarization_data()
    topic_descs = {k: TOPICS_GSS.get(k, k) for k in SELECTED_TOPICS}

    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(config["path"])
        model_type = config["type"]

        for i, (topic, (category, mismatch_type)) in enumerate(SELECTED_TOPICS.items()):
            topic_desc = topic_descs[topic]

            # Generate standard prompts
            if model_type == "base":
                template = BASE_TEMPLATES.get(category, "{name} discusses {topic}:")
                prompts = [template.format(name=n, topic=topic_desc) for n in politician_names]
                system_msg = ""
            else:
                prompts = [f"Generate a statement by {n} on {topic_desc}." for n in politician_names]
                system_msg = SYSTEM_MSG_POLITICIAN

            activations = extract_heads_batched(
                model, tokenizer, prompts, system_msg,
                batch_size=config["batch_size"], max_length=MAX_LENGTH
            )

            # Per-layer Mahalanobis
            layer_dists = compute_per_layer_mahalanobis(activations, party_labels, n_components=15)
            n_layers = len(layer_dists)

            # Also compute standard (all-layer average) for reference
            N, L, H, D = activations.shape
            flat = activations.mean(axis=1).reshape(N, -1)
            n_comp = min(15, N - 1, flat.shape[1])
            from sklearn.decomposition import PCA as PCA2
            pca = PCA2(n_components=n_comp)
            reduced = pca.fit_transform(flat)
            dem_mask = party_labels == 100
            rep_mask = party_labels == 200
            cD = reduced[dem_mask].mean(axis=0)
            cR = reduced[rep_mask].mean(axis=0)
            cov_D = np.cov(reduced[dem_mask].T)
            cov_R = np.cov(reduced[rep_mask].T)
            nD, nR = dem_mask.sum(), rep_mask.sum()
            pcov = ((nD-1)*cov_D + (nR-1)*cov_R) / (nD+nR-2)
            try:
                avg_mahal = mahalanobis(cD, cR, np.linalg.pinv(pcov))
            except:
                avg_mahal = 0.0

            # GSS polarization
            pol_row = pol_data[pol_data['variable'] == topic]
            gss_pol = pol_row['polarization'].values[0] if len(pol_row) > 0 else np.nan

            result = {
                'model': model_name,
                'model_type': model_type,
                'family': config['family'],
                'topic': topic,
                'category': category,
                'mismatch_type': mismatch_type,
                'n_layers': n_layers,
                'avg_mahal': avg_mahal,
                'gss_polarization': gss_pol,
            }
            for l_idx, d in enumerate(layer_dists):
                result[f'layer_{l_idx}'] = d

            all_results.append(result)
            print(f"  [{i+1}/{len(SELECTED_TOPICS)}] {topic:12s} avg_mahal={avg_mahal:.3f} gss={gss_pol:.3f} best_layer={np.argmax(layer_dists)} (mahal={max(layer_dists):.3f})")

            del activations
            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Save checkpoint
        df_res = pd.DataFrame(all_results)
        save_checkpoint(df_res, f"{EXPERIMENT_NAME}_{model_name}", timestamp)

    # =============================================================================
    # Analysis
    # =============================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_results_{timestamp}.csv", index=False)

    # For each model, compute per-layer correlation with GSS
    print("\n--- Per-Layer Pearson r with GSS Polarization ---")
    print("(Finding which layers best predict survey polarization)")

    for model_name in df_res['model'].unique():
        df_model = df_res[df_res['model'] == model_name]
        n_layers = df_model['n_layers'].iloc[0]
        gss_vals = df_model['gss_polarization'].values

        layer_corrs = []
        for l in range(n_layers):
            layer_vals = df_model[f'layer_{l}'].values
            valid = ~np.isnan(gss_vals) & ~np.isnan(layer_vals)
            if valid.sum() >= 5:
                r, p = pearsonr(layer_vals[valid], gss_vals[valid])
                layer_corrs.append((l, r, p))
            else:
                layer_corrs.append((l, 0.0, 1.0))

        # Find best layer
        best_layer = max(layer_corrs, key=lambda x: x[1])
        worst_layer = min(layer_corrs, key=lambda x: x[1])

        # Also compute overall avg correlation
        avg_valid = ~np.isnan(gss_vals) & ~np.isnan(df_model['avg_mahal'].values)
        avg_r = pearsonr(df_model['avg_mahal'].values[avg_valid], gss_vals[avg_valid])[0] if avg_valid.sum() >= 5 else 0

        print(f"\n  {model_name}:")
        print(f"    Avg-layer r = {avg_r:.3f}")
        print(f"    Best layer: {best_layer[0]} (r={best_layer[1]:.3f}, p={best_layer[2]:.4f})")
        print(f"    Worst layer: {worst_layer[0]} (r={worst_layer[1]:.3f})")

        # Show layer profile (quartiles)
        corr_vals = [c[1] for c in layer_corrs]
        q1 = n_layers // 4
        q2 = n_layers // 2
        q3 = 3 * n_layers // 4
        print(f"    Q1 (layers 0-{q1-1}): mean_r = {np.mean(corr_vals[:q1]):.3f}")
        print(f"    Q2 (layers {q1}-{q2-1}): mean_r = {np.mean(corr_vals[q1:q2]):.3f}")
        print(f"    Q3 (layers {q2}-{q3-1}): mean_r = {np.mean(corr_vals[q2:q3]):.3f}")
        print(f"    Q4 (layers {q3}-{n_layers-1}): mean_r = {np.mean(corr_vals[q3:]):.3f}")

        # Print all layer correlations
        print(f"    All layers: ", end="")
        for l, r, p in layer_corrs:
            marker = " *" if r == best_layer[1] else ""
            print(f"L{l}={r:.2f}{marker}", end="  ")
            if (l + 1) % 8 == 0:
                print(f"\n{'':16s}", end="")
        print()

    # Compare over-polarized vs under-polarized topics: different layer profiles?
    print(f"\n--- Layer Profile by Mismatch Type ---")
    for model_name in df_res['model'].unique():
        df_model = df_res[df_res['model'] == model_name]
        n_layers = df_model['n_layers'].iloc[0]

        print(f"\n  {model_name}:")
        for mtype in ['over', 'under', 'aligned_high', 'aligned_low']:
            sub = df_model[df_model['mismatch_type'] == mtype]
            if len(sub) == 0:
                continue
            # Average layer profile for this mismatch type
            profile = []
            for l in range(n_layers):
                profile.append(sub[f'layer_{l}'].mean())

            peak_layer = np.argmax(profile)
            print(f"    {mtype:15s}: peak_layer={peak_layer}, peak_mahal={profile[peak_layer]:.3f}, "
                  f"early_mean={np.mean(profile[:n_layers//2]):.3f}, late_mean={np.mean(profile[n_layers//2:]):.3f}")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 0B COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
