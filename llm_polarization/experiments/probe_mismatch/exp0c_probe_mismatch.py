"""
Experiment 0C: Probe-Based Mismatch Analysis

Research Question:
    Does linear probe accuracy / confidence correlate with GSS polarization
    better than Mahalanobis distance? Probes learn the optimal decision boundary;
    their confidence margin may capture "how opinionated" the model is better
    than raw centroid separation.

Method:
    For 20 selected topics:
    - Extract (N, L, H, D) activations, average across layers → (N, H*D), PCA-reduce
    - Train 5-fold CV linear probes (D vs R classification)
    - Compute: accuracy, AUC, mean confidence (P(correct class) averaged over test set)
    - Also compute confidence VARIANCE per topic (do some politicians get ambiguous signals?)
    - Correlate each metric with GSS polarization
    - Compare: which metric best predicts survey outcomes?

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr

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

EXPERIMENT_NAME = "exp0c"
MAX_LENGTH = 128
POLITICIAN_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"

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
# Probe Analysis
# =============================================================================

def compute_probe_metrics(activations: np.ndarray, labels: np.ndarray, n_components: int = 15, n_folds: int = 5):
    """
    Train 5-fold CV linear probe and compute multiple metrics.

    Returns dict with: accuracy, auc, mean_confidence, confidence_std, margin_mean, margin_std
    """
    N, L, H, D = activations.shape
    flat = activations.mean(axis=1).reshape(N, -1)

    n_comp = min(n_components, N - 1, flat.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(flat)

    # Binary labels: 100 → 0 (Dem), 200 → 1 (Rep)
    y = (labels == 200).astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_acc = []
    all_auc = []
    all_confidences = []  # P(correct class) for each test sample
    all_margins = []      # |P(rep) - P(dem)| for each test sample

    for train_idx, test_idx in skf.split(reduced, y):
        X_train, X_test = reduced[train_idx], reduced[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        except:
            auc = 0.5

        # Confidence: P(correct class) for each sample
        for i, (true_label, proba) in enumerate(zip(y_test, y_proba)):
            confidence = proba[true_label]  # P(correct class)
            margin = abs(proba[1] - proba[0])  # |P(rep) - P(dem)|
            all_confidences.append(confidence)
            all_margins.append(margin)

        all_acc.append(acc)
        all_auc.append(auc)

    return {
        'accuracy': np.mean(all_acc),
        'auc': np.mean(all_auc),
        'mean_confidence': np.mean(all_confidences),
        'confidence_std': np.std(all_confidences),
        'margin_mean': np.mean(all_margins),
        'margin_std': np.std(all_margins),
        # Fraction of samples with low confidence (< 0.6) — "ambiguous" politicians
        'frac_ambiguous': np.mean(np.array(all_confidences) < 0.6),
    }


def compute_mahalanobis_pca(activations: np.ndarray, labels: np.ndarray, n_components: int = 15) -> float:
    N, L, H, D = activations.shape
    flat = activations.mean(axis=1).reshape(N, -1)
    n_comp = min(n_components, N - 1, flat.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(flat)

    dem_mask = labels == 100
    rep_mask = labels == 200
    if dem_mask.sum() < 2 or rep_mask.sum() < 2:
        return 0.0

    cD = reduced[dem_mask].mean(axis=0)
    cR = reduced[rep_mask].mean(axis=0)
    cov_D = np.cov(reduced[dem_mask].T)
    cov_R = np.cov(reduced[rep_mask].T)
    nD, nR = dem_mask.sum(), rep_mask.sum()
    pcov = ((nD-1)*cov_D + (nR-1)*cov_R) / (nD+nR-2)

    try:
        return mahalanobis(cD, cR, np.linalg.pinv(pcov))
    except:
        return 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"{'='*80}")
    print(f"EXPERIMENT 0C: PROBE-BASED MISMATCH ANALYSIS")
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

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(config["path"])
        model_type = config["type"]

        for i, (topic, (category, mismatch_type)) in enumerate(SELECTED_TOPICS.items()):
            topic_desc = topic_descs[topic]

            if model_type == "base":
                template = BASE_TEMPLATES.get(category, "{name} discusses {topic}:")
                prompts = [template.format(name=n, topic=topic_desc) for n in politician_names]
                system_msg = ""
            else:
                prompts = [f"Generate a statement by {n} on {topic_desc}." for n in politician_names]
                system_msg = config.get("system_override", SYSTEM_MSG_POLITICIAN)

            activations = extract_heads_batched(
                model, tokenizer, prompts, system_msg,
                batch_size=config["batch_size"], max_length=MAX_LENGTH
            )

            # Compute Mahalanobis
            mahal = compute_mahalanobis_pca(activations, party_labels, n_components=15)

            # Compute probe metrics
            probe = compute_probe_metrics(activations, party_labels, n_components=15)

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
                'mahalanobis': mahal,
                'gss_polarization': gss_pol,
                **probe,
            }
            all_results.append(result)

            print(f"  [{i+1}/{len(SELECTED_TOPICS)}] {topic:12s} mahal={mahal:.3f} acc={probe['accuracy']:.3f} "
                  f"conf={probe['mean_confidence']:.3f} margin={probe['margin_mean']:.3f} "
                  f"ambig={probe['frac_ambiguous']:.2f} gss={gss_pol:.3f}")

            del activations
            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

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

    # Compare correlation of each metric with GSS polarization
    metrics = ['mahalanobis', 'accuracy', 'auc', 'mean_confidence', 'margin_mean', 'frac_ambiguous']
    metric_labels = ['Mahalanobis', 'Probe Accuracy', 'Probe AUC', 'Mean Confidence', 'Mean Margin', 'Frac Ambiguous']

    print("\n--- Pearson r with GSS Polarization per Model × Metric ---")
    print(f"{'Model':35s}", end="")
    for ml in metric_labels:
        print(f" {ml:>15s}", end="")
    print()
    print("-" * (35 + 16 * len(metrics)))

    summary_corrs = {m: [] for m in metrics}

    for model_name in sorted(df_res['model'].unique()):
        df_model = df_res[df_res['model'] == model_name]
        gss = df_model['gss_polarization'].values
        print(f"{model_name:35s}", end="")

        for metric in metrics:
            vals = df_model[metric].values
            valid = ~np.isnan(gss) & ~np.isnan(vals)
            if valid.sum() >= 5:
                r, _ = pearsonr(vals[valid], gss[valid])
                summary_corrs[metric].append(r)
                print(f" {r:15.3f}", end="")
            else:
                print(f" {'N/A':>15s}", end="")
        print()

    # Summary across models
    print(f"\n{'Mean across models':35s}", end="")
    for metric in metrics:
        if summary_corrs[metric]:
            print(f" {np.mean(summary_corrs[metric]):15.3f}", end="")
        else:
            print(f" {'N/A':>15s}", end="")
    print()

    # By mismatch type: do probes disagree with Mahalanobis?
    print(f"\n--- Probe Accuracy vs Mahalanobis by Mismatch Type ---")
    for mtype in ['over', 'under', 'aligned_high', 'aligned_low']:
        sub = df_res[df_res['mismatch_type'] == mtype]
        print(f"\n  {mtype} (n={len(sub)}):")
        print(f"    Mahalanobis: mean={sub['mahalanobis'].mean():.3f}, std={sub['mahalanobis'].std():.3f}")
        print(f"    Accuracy:    mean={sub['accuracy'].mean():.3f}, std={sub['accuracy'].std():.3f}")
        print(f"    Confidence:  mean={sub['mean_confidence'].mean():.3f}, std={sub['mean_confidence'].std():.3f}")
        print(f"    Ambiguous:   mean={sub['frac_ambiguous'].mean():.3f}, std={sub['frac_ambiguous'].std():.3f}")

    # Key question: does probe give different ordering than Mahalanobis?
    print(f"\n--- Topic Ranking Comparison: Mahalanobis vs Probe Confidence ---")
    for model_name in sorted(df_res['model'].unique()):
        df_model = df_res[df_res['model'] == model_name].copy()
        df_model['rank_mahal'] = df_model['mahalanobis'].rank(ascending=False)
        df_model['rank_conf'] = df_model['mean_confidence'].rank(ascending=False)
        df_model['rank_diff'] = df_model['rank_mahal'] - df_model['rank_conf']

        biggest_diff = df_model.reindex(df_model['rank_diff'].abs().nlargest(5).index)
        rho, _ = pearsonr(df_model['rank_mahal'], df_model['rank_conf'])
        print(f"\n  {model_name} (rank correlation = {rho:.3f}):")
        for _, row in biggest_diff.iterrows():
            print(f"    {row['topic']:12s} mahal_rank={row['rank_mahal']:2.0f} conf_rank={row['rank_conf']:2.0f} "
                  f"diff={row['rank_diff']:+3.0f} ({row['mismatch_type']})")

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 0C COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
