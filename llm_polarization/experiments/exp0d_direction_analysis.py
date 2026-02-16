"""
Experiment 0D: D-R Activation Direction Consistency

Research Question:
    Do over-polarized and under-polarized topics use different activation
    dimensions for party separation? Or does the model use a single "partisan
    axis" for all topics, just with varying magnitude?

Intuition:
    If there's a universal partisan axis → the mismatch is purely about magnitude
    (how much the LLM activates that axis per topic).
    If different topics use different axes → the mismatch is structural
    (the model encodes different topics in fundamentally different ways).

Method:
    For 20 selected topics:
    - Extract activations, PCA-reduce to (N, K)
    - Compute D-R direction vector: centroid_R - centroid_D (K-dim)
    - Normalize to unit vector
    - Compute pairwise cosine similarity between direction vectors across topics
    - Cluster topics by their direction vectors
    - Compare: do over/under-polarized topics cluster differently?

Models: Qwen3-4B (3) + SmolLM3-3B (3)
Runtime: ~15 min on A100
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, mahalanobis
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering

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

EXPERIMENT_NAME = "exp0d"
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
# Direction Analysis
# =============================================================================

def extract_dr_direction(activations: np.ndarray, labels: np.ndarray, n_components: int = 15):
    """
    Extract the D-R separation direction in PCA space.

    Returns:
        direction: normalized (centroid_R - centroid_D) unit vector
        magnitude: Euclidean distance between centroids
        reduced: PCA-reduced activations (N, K)
    """
    N, L, H, D = activations.shape
    flat = activations.mean(axis=1).reshape(N, -1)

    n_comp = min(n_components, N - 1, flat.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(flat)

    dem_mask = labels == 100
    rep_mask = labels == 200

    centroid_D = reduced[dem_mask].mean(axis=0)
    centroid_R = reduced[rep_mask].mean(axis=0)

    direction = centroid_R - centroid_D
    magnitude = np.linalg.norm(direction)

    if magnitude > 0:
        direction_norm = direction / magnitude
    else:
        direction_norm = direction

    return direction_norm, magnitude, reduced, pca.explained_variance_ratio_


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"{'='*80}")
    print(f"EXPERIMENT 0D: D-R ACTIVATION DIRECTION CONSISTENCY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    set_random_seeds(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_pol = load_politicians(POLITICIAN_CSV)
    politician_names = df_pol['bioname'].tolist()
    party_labels = df_pol['party_code'].values
    pol_data = load_polarization_data()
    topic_descs = {k: TOPICS_GSS.get(k, k) for k in SELECTED_TOPICS}

    all_results = []
    # Store direction vectors per model for pairwise comparison
    model_directions = {}  # model_name -> {topic: direction_vector}

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(config["path"])
        model_type = config["type"]
        model_directions[model_name] = {}

        for i, (topic, (category, mismatch_type)) in enumerate(SELECTED_TOPICS.items()):
            topic_desc = topic_descs[topic]

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

            direction, magnitude, reduced, var_ratio = extract_dr_direction(activations, party_labels, n_components=15)
            model_directions[model_name][topic] = direction

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
                'magnitude': magnitude,
                'gss_polarization': gss_pol,
                'pca_var_explained_top3': sum(var_ratio[:3]),
            }
            all_results.append(result)

            print(f"  [{i+1}/{len(SELECTED_TOPICS)}] {topic:12s} magnitude={magnitude:.3f} gss={gss_pol:.3f} "
                  f"pca_var_top3={sum(var_ratio[:3]):.3f}")

            del activations, reduced
            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        df_res = pd.DataFrame(all_results)
        save_checkpoint(df_res, f"{EXPERIMENT_NAME}_{model_name}", timestamp)

    # =============================================================================
    # Analysis: Pairwise Direction Similarity
    # =============================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_results_{timestamp}.csv", index=False)

    topic_list = list(SELECTED_TOPICS.keys())
    n_topics = len(topic_list)

    for model_name in sorted(model_directions.keys()):
        dirs = model_directions[model_name]
        if len(dirs) < n_topics:
            continue

        print(f"\n{'='*60}")
        print(f"Direction Analysis: {model_name}")
        print(f"{'='*60}")

        # Build pairwise cosine similarity matrix
        sim_matrix = np.zeros((n_topics, n_topics))
        for a in range(n_topics):
            for b in range(n_topics):
                d_a = dirs[topic_list[a]]
                d_b = dirs[topic_list[b]]
                # cosine similarity = 1 - cosine distance
                sim = 1.0 - cosine(d_a, d_b) if np.linalg.norm(d_a) > 0 and np.linalg.norm(d_b) > 0 else 0.0
                sim_matrix[a, b] = sim

        # Print similarity matrix (abbreviated)
        print(f"\n  Pairwise Cosine Similarity (direction vectors):")
        print(f"  {'':12s}", end="")
        for t in topic_list:
            print(f" {t[:8]:>8s}", end="")
        print()
        for a in range(n_topics):
            print(f"  {topic_list[a]:12s}", end="")
            for b in range(n_topics):
                print(f" {sim_matrix[a,b]:8.3f}", end="")
            print()

        # Average within-group vs between-group similarity
        mismatch_types = [SELECTED_TOPICS[t][1] for t in topic_list]
        print(f"\n  Within-group vs Between-group Direction Similarity:")

        groups = ['over', 'under', 'aligned_high', 'aligned_low']
        for g in groups:
            g_idx = [i for i, t in enumerate(mismatch_types) if t == g]
            within = []
            between = []
            for i in g_idx:
                for j in g_idx:
                    if i != j:
                        within.append(sim_matrix[i, j])
                for j in range(n_topics):
                    if j not in g_idx:
                        between.append(sim_matrix[i, j])

            w_mean = np.mean(within) if within else 0
            b_mean = np.mean(between) if between else 0
            print(f"    {g:15s}: within={w_mean:.3f}, between={b_mean:.3f}, diff={w_mean-b_mean:+.3f}")

        # Key question: is there a "universal partisan axis"?
        # Check: average cosine similarity across all topic pairs
        upper_tri = sim_matrix[np.triu_indices(n_topics, k=1)]
        print(f"\n  Overall direction consistency: mean_sim={np.mean(upper_tri):.3f}, std={np.std(upper_tri):.3f}")
        print(f"  If mean_sim > 0.7: strong universal partisan axis")
        print(f"  If mean_sim < 0.3: topic-specific encoding")

        # Cluster topics by direction
        dist_matrix = 1 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        clustering = AgglomerativeClustering(n_clusters=4, metric='precomputed', linkage='average')
        cluster_labels = clustering.fit_predict(dist_matrix)

        print(f"\n  Hierarchical Clustering (4 clusters):")
        for c in range(4):
            members = [topic_list[i] for i in range(n_topics) if cluster_labels[i] == c]
            member_types = [SELECTED_TOPICS[m][1] for m in members]
            print(f"    Cluster {c}: {members}")
            print(f"             types: {member_types}")

    # Magnitude vs GSS correlation
    print(f"\n--- Euclidean Magnitude vs GSS Polarization ---")
    for model_name in sorted(df_res['model'].unique()):
        df_model = df_res[df_res['model'] == model_name]
        valid = ~np.isnan(df_model['gss_polarization'].values)
        if valid.sum() >= 5:
            r, p = pearsonr(df_model['magnitude'].values[valid], df_model['gss_polarization'].values[valid])
            print(f"  {model_name:35s}: r={r:.3f} (p={p:.4f})")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 0D COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
