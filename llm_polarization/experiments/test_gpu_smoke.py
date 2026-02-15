"""
GPU smoke test: run 1 model on 1 topic to validate the full GPU pipeline.
Should complete in <2 minutes on any GPU.
"""
import sys
import time
import json
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from shared_utils import (
    compute_pca_and_distance, TOPIC_LISTS_DIR, MODEL_FAMILIES,
)
from model_utils import load_model, extract_heads_batched
from prompt_utils import load_politicians, generate_politician_prompts, POLITICIAN_TEMPLATES
from config import SYSTEM_MSG_POLITICIAN
from exp_bramson_dimensions import compute_all_bramson_dimensions

print("="*80)
print("GPU SMOKE TEST")
print("="*80)

# Check GPU
assert torch.cuda.is_available(), "No GPU available!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Use smallest model + 1 topic
MODEL_PATH = MODEL_FAMILIES['Qwen3-4B']['base']['path']
BATCH_SIZE = 180

# Load politicians
POL_CSV = "/project/jevans/maxzhuyt/data/HS116_members_fullname.csv"
df_pol = load_politicians(POL_CSV)
politician_names = df_pol['fullname'].tolist()
politician_labels = (df_pol['party_code'].values == 200).astype(int)
print(f"Politicians: {len(politician_names)} (D={int(np.sum(politician_labels==0))}, R={int(np.sum(politician_labels==1))})")

# Pick 1 topic
with open(TOPIC_LISTS_DIR / "exp5_polarized_topics.json") as f:
    topics = json.load(f)
topic_name = list(topics.keys())[0]
topic_desc = topics[topic_name]
print(f"Topic: {topic_name} -> {topic_desc[:60]}...")

# ===== TEST 1: Base model (completion-style) =====
print("\n--- Test 1: Base model, completion-style ---")
t0 = time.time()

model, tokenizer = load_model(MODEL_PATH)
tokenizer.chat_template = None

BASE_TEMPLATE = "{name} makes a statement on {topic}:"
prompts = [BASE_TEMPLATE.format(name=n, topic=topic_desc) for n in politician_names]
print(f"  Example prompt: {prompts[0]}")
print(f"  Num prompts: {len(prompts)}")

X = extract_heads_batched(model, tokenizer, prompts, "",
                          batch_size=BATCH_SIZE, max_length=128)
print(f"  Activations shape: {X.shape}")
assert X.shape[0] == len(politician_names), f"Expected {len(politician_names)} rows, got {X.shape[0]}"
assert X.shape[1] > 0, "Zero features"
assert not np.isnan(X).any(), "NaN in activations"
assert not np.isinf(X).any(), "Inf in activations"

# PCA + Mahalanobis
pca_result = compute_pca_and_distance(X, politician_labels, pca_dim=15)
print(f"  PC1 variance explained: {pca_result['variance_explained']:.4f}")
print(f"  Mahalanobis distance: {pca_result['mahalanobis_dist']:.4f}")

# Bramson dimensions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X_flat = X.reshape(X.shape[0], -1) if X.ndim > 2 else X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)
pca = PCA(n_components=1)
data_1d = pca.fit_transform(X_scaled).flatten()
dims = compute_all_bramson_dimensions(data_1d, politician_labels)
print(f"  Bramson dimensions: {', '.join(f'{k}={v:.3f}' for k,v in dims.items())}")

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
print(f"  Time: {time.time()-t0:.1f}s")

# ===== TEST 2: Instruct model (chat-style) =====
print("\n--- Test 2: Instruct model, chat-style ---")
t0 = time.time()

INSTRUCT_PATH = MODEL_FAMILIES['Qwen3-4B']['instruct']['path']
model, tokenizer = load_model(INSTRUCT_PATH)

template = POLITICIAN_TEMPLATES['default']
prompts = generate_politician_prompts(topic_desc, politician_names, template=template)
print(f"  Example prompt: {prompts[0]}")
print(f"  System msg: {SYSTEM_MSG_POLITICIAN[:60]}...")

X2 = extract_heads_batched(model, tokenizer, prompts, SYSTEM_MSG_POLITICIAN,
                           batch_size=BATCH_SIZE, max_length=128)
print(f"  Activations shape: {X2.shape}")
assert X2.shape[0] == len(politician_names)
assert not np.isnan(X2).any(), "NaN in instruct activations"

pca_result2 = compute_pca_and_distance(X2, politician_labels, pca_dim=15)
print(f"  PC1 variance explained: {pca_result2['variance_explained']:.4f}")
print(f"  Mahalanobis distance: {pca_result2['mahalanobis_dist']:.4f}")

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()
print(f"  Time: {time.time()-t0:.1f}s")

# ===== Summary =====
print("\n" + "="*80)
print("GPU SMOKE TEST PASSED")
print("="*80)
print(f"Base model activations: {X.shape}")
print(f"Instruct model activations: {X2.shape}")
print(f"Mahalanobis (base): {pca_result['mahalanobis_dist']:.4f}")
print(f"Mahalanobis (instruct): {pca_result2['mahalanobis_dist']:.4f}")
print("Both models loaded, ran inference, extracted activations, computed stats.")
print("Safe to submit full experiment jobs.")
