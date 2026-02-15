# Significant Correlations between Polarization CoG and Fundamentalness Measures

*31 significant measures out of 89 tested (p < 0.05 for Spearman ρ)*

---

## Data Processing

**Source**: General Social Survey (GSS) 2021-2024 waves

**Row filtering**:
- Respondents with valid responses on 1-7 ordinal scales
- Invalid codes excluded: 0 (not applicable), 8 (don't know), 9 (no answer), negative values

**Column filtering**:
- Questions appearing in ≥2 survey years retained
- Questions with ≥100 valid responses retained
- Duplicate ballot versions merged (e.g., natcrime + natcrimy → natcrime)
- **Excluded**: `polviews` (self-reported ideology), `partyid` (party identification)
- **Final sample**: 97 questions, N=82 with both CoG and fundamentalness data

---

## Results Table

| Measure | N | Pearson *r* | *p* | Spearman *ρ* | *p* | Procedure |
|:--------|:--:|:-----------:|:---:|:------------:|:---:|:----------|
| mi_max | 82 | +0.288** | 0.009 | +0.395*** | <.001 | Maximum pairwise MI for each question |
| orig_MI | 82 | +0.286** | 0.009 | +0.383*** | <.001 | Composite MI = normalized sum of pairwise MI |
| mi_sum | 82 | +0.258* | 0.019 | +0.355** | 0.001 | Sum of pairwise MI with all other questions |
| mi_mean | 82 | +0.258* | 0.019 | +0.355** | 0.001 | Mean pairwise MI with all other questions |
| tree_max_neighbor_mi_hierarchy | 82 | +0.318** | 0.004 | +0.352** | 0.001 | Chow-Liu MST, root=max neighbor MI, hierarchy score |
| tree_max_neighbor_mi_depth_inv | 82 | +0.318** | 0.004 | +0.352** | 0.001 | Chow-Liu MST, root=max neighbor MI, inverse depth |
| tree_max_neighbor_mi_depth_norm | 82 | +0.318** | 0.004 | +0.352** | 0.001 | Chow-Liu MST, root=max neighbor MI, normalized depth |
| orig_Tree | 82 | +0.339** | 0.002 | +0.350** | 0.001 | Composite tree: depth + subtree size + betweenness |
| thresh_0.2_pagerank | 82 | +0.274* | 0.013 | +0.350** | 0.001 | Sparse network (\|corr\|>0.2), PageRank |
| complete_squared_harmonic | 82 | +0.226* | 0.041 | +0.344** | 0.002 | Complete network, weight=corr², harmonic centrality |
| tree_max_degree_depth_inv | 82 | +0.284** | 0.010 | +0.307** | 0.005 | Chow-Liu MST, root=max degree, inverse depth |
| tree_max_degree_hierarchy | 82 | +0.284** | 0.010 | +0.307** | 0.005 | Chow-Liu MST, root=max degree, hierarchy score |
| tree_max_degree_depth_norm | 82 | +0.284** | 0.010 | +0.307** | 0.005 | Chow-Liu MST, root=max degree, normalized depth |
| complete_squared_closeness | 82 | +0.218* | 0.049 | +0.296** | 0.007 | Complete network, weight=corr², closeness |
| complete_squared_strength | 82 | +0.215 | 0.052 | +0.296** | 0.007 | Complete network, weight=corr², strength |
| complete_harmonic | 82 | +0.194 | 0.081 | +0.295** | 0.007 | Complete network, weight=\|corr\|, harmonic |
| complete_squared_katz | 82 | +0.216 | 0.051 | +0.289** | 0.009 | Complete network, weight=corr², Katz centrality |
| complete_squared_pagerank | 82 | +0.210 | 0.058 | +0.284** | 0.010 | Complete network, weight=corr², PageRank |
| complete_log_harmonic | 82 | +0.180 | 0.106 | +0.282* | 0.010 | Complete network, distance=-log\|corr\|, harmonic |
| orig_Predictive | 82 | +0.197 | 0.076 | +0.274* | 0.013 | Mean R² from linear models predicting all others |
| complete_squared_eigenvector | 82 | +0.195 | 0.080 | +0.260* | 0.018 | Complete network, weight=corr², eigenvector |
| thresh_0.2_harmonic | 82 | +0.036 | 0.750 | +0.250* | 0.023 | Sparse network (\|corr\|>0.2), harmonic |
| thresh_0.05_harmonic | 82 | +0.140 | 0.211 | +0.249* | 0.024 | Sparse network (\|corr\|>0.05), harmonic |
| orig_Network | 82 | +0.174 | 0.117 | +0.248* | 0.025 | Composite: betweenness + closeness + eigenvector |
| thresh_0.2_strength | 82 | +0.189 | 0.088 | +0.242* | 0.028 | Sparse network (\|corr\|>0.2), strength |
| thresh_0.2_katz | 82 | +0.185 | 0.097 | +0.241* | 0.029 | Sparse network (\|corr\|>0.2), Katz |
| thresh_0.2_eigenvector | 82 | +0.182 | 0.101 | +0.231* | 0.037 | Sparse network (\|corr\|>0.2), eigenvector |
| thresh_0.15_pagerank | 82 | +0.170 | 0.127 | +0.227* | 0.040 | Sparse network (\|corr\|>0.15), PageRank |
| thresh_0.15_harmonic | 82 | +0.138 | 0.217 | +0.222* | 0.045 | Sparse network (\|corr\|>0.15), harmonic |
| tree_max_degree_depth | 82 | -0.284** | 0.010 | -0.307** | 0.005 | Chow-Liu MST, root=max degree, raw depth |
| tree_max_neighbor_mi_depth | 82 | -0.318** | 0.004 | -0.352** | 0.001 | Chow-Liu MST, root=max neighbor MI, raw depth |

*\*p < .05, \*\*p < .01, \*\*\*p < .001*

---

## Method Descriptions

### Tree-Based Measures (Chow-Liu)
1. **Build maximum spanning tree** from pairwise mutual information matrix (sklearn `mutual_info_score`)
2. **Select root** using one of:
   - `max_neighbor_mi`: Node with highest sum of MI to its immediate neighbors
   - `max_degree`: Node with most connections in MST
3. **Compute depth** = shortest path length from root to each node
4. **Transform**:
   - `hierarchy` / `depth_norm` = 1 - (depth / max_depth), so root=1, leaves→0
   - `depth_inv` = max_depth - depth
   - `depth` = raw depth (root=0), hence negative correlation with CoG

### MI-Based Measures
1. **Compute pairwise MI** using sklearn's `mutual_info_score` on ordinal responses (treated as discrete)
2. **Aggregate per question**:
   - `mi_max`: Maximum MI with any single other question (strongest connection)
   - `mi_sum`: Sum of MI with all other questions (total shared information)
   - `mi_mean`: Mean MI (average information per pair)

### Network Measures
1. **Build graph** from pairwise Spearman correlations:
   - `complete`: All question pairs connected
   - `thresh_X`: Only edges where |correlation| > X
2. **Edge weights**:
   - Standard: weight = |correlation|, distance = 1 - |correlation|
   - `squared`: weight = correlation², distance = 1 - correlation²
   - `log`: distance = -log(|correlation|)
3. **Centrality measures** (NetworkX):
   - `harmonic`: Sum of inverse distances (handles disconnected graphs)
   - `closeness`: Inverse of average shortest path to all other nodes
   - `eigenvector`: Importance based on connections to important nodes
   - `pagerank`: Iterative importance (random walk probability)
   - `katz`: Influence accounting for all paths, attenuated by length
   - `strength`: Sum of edge weights (weighted degree)

### Predictive Power
1. For each question Q, fit linear regression: Q → all other questions
2. Compute R² for each prediction
3. Score = mean R² across all predictions (how well Q predicts the belief system)

### Original Composite Scores
- `orig_MI`: Normalized sum of pairwise MI
- `orig_Tree`: Combined tree depth, subtree size, and betweenness
- `orig_Network`: Combined betweenness, closeness, and eigenvector centrality
- `orig_Predictive`: Mean R² from linear predictions
