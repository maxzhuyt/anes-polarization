# Fundamentalness vs Center of Gravity: Per-Model Correlations (Public Issues)

## Overview

This table shows **Spearman rank correlations (ρ)** between fundamentalness measures computed from GSS survey data and the Center of Gravity (CoG) of polarization signals in LLM hidden layers.

- **Fundamentalness**: Measures of how "central" or "foundational" a question is within the belief network (computed from GSS 2021-2024 survey responses)
- **Center of Gravity (CoG)**: The weighted average layer index where polarization signal appears in the LLM (higher = deeper encoding)

## Key Findings

- **Mistral-7B-v0.2** shows the strongest correlations (ρ up to +0.678***)
- **Gemma-2-9b** shows consistent moderate-strong correlations (ρ up to +0.477***)
- **Qwen3-8B** shows weaker but mostly significant correlations
- **Llama-3.1-8B** shows the weakest correlations (mostly non-significant)

## Per-Model Spearman ρ (Public Issues, N=126 questions)

| Measure | Llama-3.1-8B | Gemma-2-9b | Mistral-7B-v0.2 | Qwen3-8B | Avg |
|---------|-------------|------------|-----------------|----------|-----|
| thresh_0.1_harmonic | +0.099 | +0.462*** | +0.637*** | +0.237** | +0.359 |
| thresh_0.2_closeness | +0.066 | +0.477*** | +0.678*** | +0.209* | +0.357 |
| thresh_0.1_closeness | +0.093 | +0.456*** | +0.637*** | +0.241** | +0.357 |
| thresh_0.2_harmonic | +0.074 | +0.468*** | +0.662*** | +0.217* | +0.355 |
| complete_eigenvector | +0.072 | +0.472*** | +0.640*** | +0.215* | +0.350 |
| thresh_0.1_strength | +0.086 | +0.454*** | +0.620*** | +0.226* | +0.346 |
| complete_harmonic | +0.103 | +0.437*** | +0.604*** | +0.235** | +0.345 |
| thresh_0.2_strength | +0.101 | +0.437*** | +0.600*** | +0.237** | +0.344 |
| mi_max | +0.260** | +0.314*** | +0.413*** | +0.374*** | +0.340 |
| complete_closeness | +0.079 | +0.432*** | +0.590*** | +0.220* | +0.330 |
| thresh_0.1_pagerank | +0.087 | +0.417*** | +0.585*** | +0.230** | +0.330 |
| complete_strength | +0.080 | +0.430*** | +0.588*** | +0.218* | +0.329 |
| mi_mean | +0.025 | +0.455*** | +0.613*** | +0.207* | +0.325 |
| mi_sum | +0.025 | +0.455*** | +0.613*** | +0.207* | +0.325 |
| complete_pagerank | +0.081 | +0.417*** | +0.573*** | +0.221* | +0.323 |
| corr_abs_mean | +0.076 | +0.422*** | +0.577*** | +0.215* | +0.323 |
| corr_abs_sum | +0.076 | +0.422*** | +0.577*** | +0.215* | +0.323 |
| thresh_0.2_pagerank | +0.173 | +0.333*** | +0.452*** | +0.262** | +0.305 |
| complete_log_betweenness | +0.167 | +0.110 | +0.161 | +0.329*** | +0.192 |
| tree_center_hierarchy | -0.078 | +0.346*** | +0.436*** | +0.049 | +0.188 |
| thresh_0.1_betweenness | +0.147 | +0.114 | +0.169 | +0.297*** | +0.182 |
| thresh_0.2_betweenness | +0.138 | +0.133 | +0.199* | +0.240** | +0.178 |
| tree_betweenness | +0.080 | +0.056 | +0.217* | +0.180* | +0.133 |
| tree_center_subtree | +0.081 | +0.055 | +0.217* | +0.180* | +0.133 |
| tree_max_degree_subtree | +0.080 | +0.054 | +0.216* | +0.181* | +0.133 |
| tree_max_neighbor_mi_subtree | +0.080 | +0.054 | +0.216* | +0.181* | +0.133 |
| complete_betweenness | +0.043 | +0.148 | +0.184* | +0.054 | +0.107 |
| tree_max_neighbor_mi_hierarchy | -0.245** | +0.152 | +0.355*** | -0.069 | +0.048 |
| tree_max_degree_hierarchy | -0.245** | +0.152 | +0.355*** | -0.069 | +0.048 |
| tree_max_degree_depth | +0.245** | -0.152 | -0.355*** | +0.069 | -0.048 |
| tree_max_neighbor_mi_depth | +0.245** | -0.152 | -0.355*** | +0.069 | -0.048 |
| tree_center_depth | +0.078 | -0.346*** | -0.436*** | -0.049 | -0.188 |

**Significance levels:** \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001

## Interpretation

**Positive correlation** means: Questions that are more "fundamental" (central in the belief network) have their polarization signals encoded in **deeper** layers of the LLM.

This suggests that:
1. LLMs encode foundational political beliefs at deeper, more abstract layers
2. Peripheral/specific beliefs are encoded at shallower layers
3. The effect is consistent across Gemma, Mistral, and Qwen, but weak in Llama

## Methods

- **Fundamentalness measures**: Computed from pairwise correlations and mutual information between GSS survey questions
- **Network centrality**: Built correlation networks with various thresholds (0.05, 0.1, 0.15, 0.2) and computed centrality metrics
- **Tree measures**: Built Chow-Liu maximum spanning tree from MI matrix with different root selections
- **CoG**: Center of gravity of Mahalanobis distance (polarization signal) across LLM layers
