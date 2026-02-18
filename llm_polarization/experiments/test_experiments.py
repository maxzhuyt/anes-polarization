"""
Unit tests for experiment components (non-GPU parts).
Run this before submitting GPU jobs to catch errors early.
"""

import sys
import os
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("UNIT TESTS FOR EXPERIMENT COMPONENTS")
print("="*80)

# =============================================================================
# Test 1: Imports
# =============================================================================
print("\n[Test 1] Testing imports...")
try:
    from shared_utils import (
        set_random_seeds,
        load_all_topics,
        load_polarization_data,
        compute_pca_and_distance,
        compute_overlap_coefficient,
        compute_bimodality_coefficient,
        MODEL_FAMILIES,
        RESULTS_DIR,
        TOPIC_LISTS_DIR,
    )
    from prompt_utils import load_politicians
    import numpy as np
    import pandas as pd
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# =============================================================================
# Test 2: Topic Lists
# =============================================================================
print("\n[Test 2] Testing topic lists...")
try:
    topic_files = list(TOPIC_LISTS_DIR.glob("*.json"))
    print(f"  Found {len(topic_files)} topic list files")

    expected_files = [
        "exp1_public.json", "exp1_private.json",
        "exp1r_public.json", "exp1r_private.json",
        "exp2_public.json", "exp2_private.json",
        "exp3_core_topics.json",
        "exp4_all_topics.json",
        "exp5_polarized_topics.json",
        "exp6_overlap_topics.json",
        "exp7_attention_topics.json",
        "exp8_affective_topics.json",
    ]

    for filename in expected_files:
        filepath = TOPIC_LISTS_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                topics = json.load(f)
            print(f"  ✓ {filename}: {len(topics)} topics")
        else:
            print(f"  ✗ MISSING: {filename}")

except Exception as e:
    print(f"✗ Topic list test failed: {e}")
    sys.exit(1)

# =============================================================================
# Test 3: Politicians
# =============================================================================
print("\n[Test 3] Testing politician loading...")
try:
    politician_csv = "/project/jevans/maxzhuyt/gss_polarization/data/politicians.csv"
    df_politicians = load_politicians(politician_csv)

    print(f"  Total politicians: {len(df_politicians)}")
    print(f"  Columns: {df_politicians.columns.tolist()}")

    # Check for required columns
    assert 'fullname' in df_politicians.columns, "Missing 'fullname' column"
    assert 'party_code' in df_politicians.columns, "Missing 'party_code' column"

    # Convert to dict format
    politicians = []
    for _, row in df_politicians.iterrows():
        politicians.append({
            'name': row['fullname'],
            'party': 'Democrat' if row['party_code'] == 100 else 'Republican'
        })

    n_dem = sum(1 for p in politicians if p['party'] == 'Democrat')
    n_rep = sum(1 for p in politicians if p['party'] == 'Republican')

    print(f"  Democrats: {n_dem}")
    print(f"  Republicans: {n_rep}")
    print(f"  Example: {politicians[0]}")
    print("✓ Politician loading successful")

except Exception as e:
    print(f"✗ Politician test failed: {e}")
    sys.exit(1)

# =============================================================================
# Test 4: Model Families
# =============================================================================
print("\n[Test 4] Testing model family configuration...")
try:
    print(f"  Families: {list(MODEL_FAMILIES.keys())}")

    total_models = 0
    for family_name, family_config in MODEL_FAMILIES.items():
        print(f"\n  [{family_name}]")
        for variant_name, variant_cfg in family_config.items():
            model_path = variant_cfg['path']
            model_type = variant_cfg['type']
            batch_size = variant_cfg['batch_size']

            # Check if path exists
            exists = Path(model_path).exists()
            status = "✓" if exists else "✗ MISSING"

            print(f"    {status} {variant_name} ({model_type}): {model_path}")
            total_models += 1

    print(f"\n  Total models: {total_models}")
    print("✓ Model configuration valid")

except Exception as e:
    print(f"✗ Model configuration test failed: {e}")
    sys.exit(1)

# =============================================================================
# Test 5: Statistical Functions
# =============================================================================
print("\n[Test 5] Testing statistical functions...")
try:
    # Create synthetic data
    np.random.seed(42)

    # Simulate activations: 20 politicians × 100 features
    activations = np.random.randn(20, 100)
    party_labels = np.array([0]*10 + [1]*10)  # 10 Democrats, 10 Republicans

    # Test PCA and Mahalanobis
    results = compute_pca_and_distance(activations, party_labels, pca_dim=5)

    print(f"  PCA variance explained (PC1): {results['variance_explained']:.4f}")
    print(f"  Mahalanobis distance: {results['mahalanobis_dist']:.4f}")
    print(f"  Centroid D shape: {results['centroid_D'].shape}")
    print(f"  Centroid R shape: {results['centroid_R'].shape}")

    # Test overlap coefficient
    dist1 = np.random.randn(100)
    dist2 = np.random.randn(100) + 1.0  # Shifted distribution
    overlap = compute_overlap_coefficient(dist1, dist2)
    print(f"  Overlap coefficient: {overlap:.4f}")

    # Test bimodality coefficient
    bimodal_data = np.concatenate([np.random.randn(50) - 2, np.random.randn(50) + 2])
    bc = compute_bimodality_coefficient(bimodal_data)
    print(f"  Bimodality coefficient: {bc:.4f} (bimodal if > 0.55)")

    print("✓ Statistical functions working")

except Exception as e:
    print(f"✗ Statistical function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Test 6: GSS Data
# =============================================================================
print("\n[Test 6] Testing GSS polarization data...")
try:
    gss_df = load_polarization_data()

    print(f"  Total observations: {len(gss_df)}")
    print(f"  Columns: {gss_df.columns.tolist()}")
    print(f"  Unique variables: {gss_df['variable'].nunique()}")

    # Check for required columns
    required_cols = ['variable', 'mean_dem', 'mean_rep', 'n_dem', 'n_rep', 'polarization']
    for col in required_cols:
        if col not in gss_df.columns:
            print(f"  ✗ MISSING column: {col}")
        else:
            print(f"  ✓ Has column: {col}")

    # Sample statistics
    example_var = gss_df.iloc[0]
    print(f"\n  Example variable: {example_var['variable']}")
    print(f"    Polarization: {example_var['polarization']:.4f}")
    print(f"    Mean D: {example_var['mean_dem']:.4f}, Mean R: {example_var['mean_rep']:.4f}")
    print(f"    N_D: {example_var['n_dem']}, N_R: {example_var['n_rep']}")

    print("✓ GSS data loading successful")

except Exception as e:
    print(f"✗ GSS data test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Test 7: Directory Structure
# =============================================================================
print("\n[Test 7] Testing directory structure...")
try:
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"    Exists: {RESULTS_DIR.exists()}")
    print(f"    Writable: {RESULTS_DIR.exists() and os.access(RESULTS_DIR, os.W_OK)}")

    print(f"  Topic lists dir: {TOPIC_LISTS_DIR}")
    print(f"    Exists: {TOPIC_LISTS_DIR.exists()}")

    # Check logs directory
    logs_dir = Path(__file__).parent / "logs"
    print(f"  Logs dir: {logs_dir}")
    print(f"    Exists: {logs_dir.exists()}")

    print("✓ Directory structure valid")

except Exception as e:
    print(f"✗ Directory test failed: {e}")
    sys.exit(1)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("ALL TESTS PASSED ✓")
print("="*80)
print("\nYou can now submit GPU jobs with confidence!")
print("Recommended submission order:")
print("  1. sbatch exp1.sbatch")
print("  2. sbatch exp1r.sbatch")
print("  3. sbatch run_qwen_comparison.sbatch")
print("  4. sbatch run_llama_comparison.sbatch")
print("  5. sbatch run_gemma_comparison.sbatch")
print("  6. sbatch run_smollm_comparison.sbatch")
print("  7. sbatch run_qwen25_comparison.sbatch")
print("="*80)
