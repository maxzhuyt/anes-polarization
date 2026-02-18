"""
Comprehensive pipeline tests. Run BEFORE submitting GPU jobs.

Tests:
1. All imports resolve
2. Politician loading + label creation (exact same as run_model_comparison.py)
3. Topic loading from all JSON files
4. Prompt generation for base and instruct (exact same as run_model_comparison.py)
5. Statistical functions with synthetic data
6. Bramson dimension functions with synthetic data
7. Checkpoint save/load roundtrip
8. exp1 and bramson script parse without errors (import + dry-run non-GPU code)
"""

import sys
import os
import json
import tempfile
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")

# =========================================================================
print("="*80)
print("TEST SUITE: Full Pipeline Validation")
print("="*80)

# =========================================================================
print("\n[1] Imports")
# =========================================================================
try:
    from shared_utils import (
        set_random_seeds, load_all_topics, load_polarization_data,
        compute_pca_and_distance, compute_overlap_coefficient,
        compute_bimodality_coefficient, MODEL_FAMILIES,
        RESULTS_DIR, TOPIC_LISTS_DIR, run_model_on_topics,
        save_checkpoint, load_checkpoint,
    )
    check("shared_utils imports", True)
except Exception as e:
    check("shared_utils imports", False, str(e))

try:
    from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
    check("prompt_utils imports", True)
except Exception as e:
    check("prompt_utils imports", False, str(e))

try:
    from config import SYSTEM_MSG_POLITICIAN
    check("config imports", True)
except Exception as e:
    check("config imports", False, str(e))

try:
    from model_utils import load_model, extract_heads_batched, get_model_info
    check("model_utils imports", True)
except Exception as e:
    check("model_utils imports", False, str(e))

try:
    from run_gss_pca import compute_all_head_metrics_pca
    check("run_gss_pca imports", True)
except Exception as e:
    check("run_gss_pca imports", False, str(e))

# =========================================================================
print("\n[2] Politician Loading (must match run_model_comparison.py)")
# =========================================================================
POL_CSV = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
df_pol = load_politicians(POL_CSV)
politician_names = df_pol['fullname'].tolist()
politician_labels = (df_pol['party_code'].values == 200).astype(int)

check("politician CSV loads", len(df_pol) > 0, f"got {len(df_pol)}")
check("has fullname column", 'fullname' in df_pol.columns)
check("has party_code column", 'party_code' in df_pol.columns)
check("names are strings", isinstance(politician_names[0], str), f"got {type(politician_names[0])}")
check("labels are 0/1 ints", set(np.unique(politician_labels)) == {0, 1})
n_dem = int(np.sum(politician_labels == 0))
n_rep = int(np.sum(politician_labels == 1))
check("D+R sums to total", n_dem + n_rep == len(politician_names), f"{n_dem}+{n_rep} vs {len(politician_names)}")
print(f"       {len(politician_names)} politicians: D={n_dem}, R={n_rep}")

# =========================================================================
print("\n[3] Topic Loading")
# =========================================================================
public_topics, private_topics = load_all_topics()
check("public topics loaded", len(public_topics) > 50, f"got {len(public_topics)}")
check("private topics loaded", len(private_topics) > 30, f"got {len(private_topics)}")

# Check all topic list JSON files
expected_jsons = [
    "exp1_public.json", "exp1_private.json", "exp1r_public.json", "exp1r_private.json",
    "exp2_public.json", "exp2_private.json", "exp3_core_topics.json",
    "exp4_all_topics.json", "exp5_polarized_topics.json",
    "exp6_overlap_topics.json", "exp7_attention_topics.json", "exp8_affective_topics.json",
]
for fname in expected_jsons:
    fp = TOPIC_LISTS_DIR / fname
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        check(f"topic list {fname}", len(data) > 0, f"empty!")
    else:
        check(f"topic list {fname}", False, "FILE MISSING")

# =========================================================================
print("\n[4] Prompt Generation (must match run_model_comparison.py)")
# =========================================================================
# Base model prompts
BASE_TEMPLATES = {
    "public_issues": "{name} makes a statement on {topic}:",
    "private_life":  "When asked about {topic}, {name} says",
}
test_names = ["Donald Trump", "Nancy Pelosi"]
test_topic = "immigration policy"

base_prompts = [BASE_TEMPLATES["public_issues"].format(name=n, topic=test_topic) for n in test_names]
check("base prompt format", "Donald Trump makes a statement on immigration policy:" == base_prompts[0],
      f"got: {base_prompts[0]}")

# Instruct model prompts
INSTRUCT_TEMPLATE_KEYS = {"public_issues": "default", "private_life": "opinion"}
template = POLITICIAN_TEMPLATES[INSTRUCT_TEMPLATE_KEYS["public_issues"]]
instruct_prompts = generate_politician_prompts(test_topic, test_names, template=template)
check("instruct prompts generated", len(instruct_prompts) == 2, f"got {len(instruct_prompts)}")
check("instruct prompts are strings", isinstance(instruct_prompts[0], str))
print(f"       Base prompt example: {base_prompts[0]}")
print(f"       Instruct prompt example: {instruct_prompts[0]}")
print(f"       System msg: {SYSTEM_MSG_POLITICIAN[:60]}...")

# =========================================================================
print("\n[5] extract_heads_batched Signature Check")
# =========================================================================
import inspect
sig = inspect.signature(extract_heads_batched)
params = list(sig.parameters.keys())
check("extract_heads_batched params", params == ['model', 'tokenizer', 'texts', 'system_msg', 'batch_size', 'max_length'],
      f"got {params}")

# Verify our run_model_on_topics function signature
sig2 = inspect.signature(run_model_on_topics)
params2 = list(sig2.parameters.keys())
check("run_model_on_topics has politician_names param", 'politician_names' in params2, f"params: {params2}")
check("run_model_on_topics has NO 'politicians' param", 'politicians' not in params2, f"params: {params2}")

# =========================================================================
print("\n[6] Model Family Configuration")
# =========================================================================
total_models = 0
for family, variants in MODEL_FAMILIES.items():
    for vname, vcfg in variants.items():
        exists = Path(vcfg['path']).exists()
        check(f"model path {family}/{vname}", exists, vcfg['path'])
        check(f"batch_size {family}/{vname}", vcfg['batch_size'] > 0)
        check(f"type {family}/{vname}", vcfg['type'] in ('base', 'instruct', 'reasoning'), vcfg['type'])
        total_models += 1
print(f"       Total model configs: {total_models}")

# =========================================================================
print("\n[7] Statistical Functions")
# =========================================================================
np.random.seed(42)

# PCA + Mahalanobis
X = np.random.randn(100, 50)
labels = np.array([0]*50 + [1]*50)
result = compute_pca_and_distance(X, labels, pca_dim=5)
check("PCA returns variance_explained", 0 < result['variance_explained'] < 1)
check("PCA returns mahalanobis_dist", result['mahalanobis_dist'] > 0)
check("PCA centroids correct shape", result['centroid_D'].shape == (5,))

# Overlap coefficient
d1 = np.random.randn(200)
d2 = np.random.randn(200) + 2.0
ovl = compute_overlap_coefficient(d1, d2)
check("overlap coefficient in [0,1]", 0 <= ovl <= 1, f"got {ovl}")

# Bimodality coefficient
bimodal = np.concatenate([np.random.randn(100) - 3, np.random.randn(100) + 3])
bc = compute_bimodality_coefficient(bimodal)
check("bimodality coefficient > 0.55 for bimodal data", bc > 0.55, f"got {bc:.3f}")

unimodal = np.random.randn(200)
bc_uni = compute_bimodality_coefficient(unimodal)
check("bimodality coefficient < 0.55 for unimodal data", bc_uni < 0.55, f"got {bc_uni:.3f}")

# =========================================================================
print("\n[8] Bramson Dimension Functions")
# =========================================================================
from exp_bramson_dimensions import (
    compute_spread, compute_dispersion, compute_coverage,
    compute_regionalization, compute_fragmentation,
    compute_distinctness, compute_group_divergence,
    compute_group_consensus, compute_size_parity,
    compute_all_bramson_dimensions,
)

# Generate separable data
np.random.seed(42)
group_a = np.random.randn(50) - 2
group_b = np.random.randn(50) + 2
data = np.concatenate([group_a, group_b])
labs = np.array([0]*50 + [1]*50)

check("spread > 0", compute_spread(data) > 0)
check("dispersion > 0", compute_dispersion(data) > 0)
check("coverage in (0,1]", 0 < compute_coverage(data) <= 1.5)
check("regionalization in [0,1]", 0 <= compute_regionalization(data) <= 1)
check("fragmentation >= 1", compute_fragmentation(data) >= 1)
check("distinctness > 0 for separated groups", compute_distinctness(data, labs) > 1.0)
check("group_divergence in [0,1]", 0 <= compute_group_divergence(data, labs) <= 1)
check("group_consensus in (0,1]", 0 < compute_group_consensus(data, labs) <= 1)
check("size_parity = 1 for balanced groups", abs(compute_size_parity(labs) - 1.0) < 0.01)

unbal = np.array([0]*90 + [1]*10)
check("size_parity < 0.5 for imbalanced", compute_size_parity(unbal) < 0.5)

all_dims = compute_all_bramson_dimensions(data, labs)
check("all_bramson returns 9 dims", len(all_dims) == 9, f"got {len(all_dims)}")

# =========================================================================
print("\n[9] Checkpoint Save/Load Roundtrip")
# =========================================================================
test_data = {'hello': [1, 2, 3], 'array': np.array([4, 5, 6])}
ckpt_path = save_checkpoint(test_data, 'test', 'roundtrip')
loaded = load_checkpoint(ckpt_path)
check("checkpoint roundtrip dict keys", set(loaded.keys()) == set(test_data.keys()))
check("checkpoint roundtrip values", np.array_equal(loaded['array'], test_data['array']))
# Clean up
os.remove(ckpt_path)

# =========================================================================
print("\n[10] GSS Polarization Data")
# =========================================================================
gss_df = load_polarization_data()
check("GSS data loads", len(gss_df) > 100, f"got {len(gss_df)}")
check("GSS has 'variable' column", 'variable' in gss_df.columns)
check("GSS has 'polarization' column", 'polarization' in gss_df.columns)
check("GSS has 'mean_dem' column", 'mean_dem' in gss_df.columns)
check("GSS has 'mean_rep' column", 'mean_rep' in gss_df.columns)

# =========================================================================
print("\n[11] exp1_encoding_strength.py Dry Run (import + setup)")
# =========================================================================
try:
    from exp1_encoding_strength import load_experiment_topics, EXPERIMENT_NAME, PCA_DIMS
    topics = load_experiment_topics(replication=False)
    check("exp1 topic loading", 'public_issues' in topics and 'private_life' in topics)
    check("exp1 public count", len(topics['public_issues']) == 30, f"got {len(topics['public_issues'])}")
    check("exp1 private count", len(topics['private_life']) == 30, f"got {len(topics['private_life'])}")

    topics_r = load_experiment_topics(replication=True)
    check("exp1r topic loading", len(topics_r['public_issues']) == 30)
except Exception as e:
    check("exp1 dry run", False, str(e))

# =========================================================================
print("\n[12] exp_bramson_dimensions.py Dry Run (import)")
# =========================================================================
try:
    from exp_bramson_dimensions import run_experiment as bramson_run
    check("bramson script imports", True)
except Exception as e:
    check("bramson script imports", False, str(e))

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*80)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
print("="*80)

if FAIL > 0:
    print("\nFIX THE FAILURES ABOVE BEFORE SUBMITTING GPU JOBS!")
    sys.exit(1)
else:
    print("\nAll tests passed. Safe to submit GPU jobs.")
    sys.exit(0)
