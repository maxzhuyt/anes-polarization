# Significant Correlations between Polarization CoG and Fundamentalness Measures

*27 significant measures out of 89 tested (p < 0.05 for Spearman ρ)*

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
- **Final sample**: 98 questions, N=89 with both CoG and fundamentalness data

---

## Results Table

| Measure | N | Pearson *r* | *p* | Spearman *ρ* | *p* | Procedure |
|:--------|:--:|:-----------:|:---:|:------------:|:---:|:----------|
| tree_max_neighbor_mi_hierarchy | 89 | +0.365*** | <.001 | +0.420*** | <.001 | Build Chow-Liu MST from MI matrix. Root = max neighbor MI sum. Score = 1 - (depth/max_depth). |
| tree_max_neighbor_mi_depth_inv | 89 | +0.365*** | <.001 | +0.420*** | <.001 | Build Chow-Liu MST from MI matrix. Root = max neighbor MI sum. Score = max_depth - depth. |
| tree_max_neighbor_mi_depth_norm | 89 | +0.365*** | <.001 | +0.420*** | <.001 | Build Chow-Liu MST from MI matrix. Root = max neighbor MI sum. Score = 1 - (depth/max_depth). |
| tree_max_degree_depth_inv | 89 | +0.342** | 0.001 | +0.392*** | <.001 | Build Chow-Liu MST from MI matrix. Root = max degree node. Score = max_depth - depth. |
| tree_max_degree_hierarchy | 89 | +0.342** | 0.001 | +0.392*** | <.001 | Build Chow-Liu MST from MI matrix. Root = max degree node. Score = 1 - (depth/max_depth). |
| tree_max_degree_depth_norm | 89 | +0.342** | 0.001 | +0.392*** | <.001 | Build Chow-Liu MST from MI matrix. Root = max degree node. Score = 1 - (depth/max_depth). |
| mi_max | 89 | +0.299** | 0.004 | +0.384*** | <.001 | Pairwise MI between all questions. Score = maximum MI value for each question. |
| orig_Tree | 89 | +0.280** | 0.008 | +0.378*** | <.001 | Original composite: tree depth + subtree size + betweenness in Chow-Liu tree. |
| orig_MI | 89 | +0.284** | 0.007 | +0.360*** | <.001 | Original composite: sum of pairwise MI with all other questions, normalized. |
| mi_sum | 89 | +0.242* | 0.022 | +0.328** | 0.002 | Pairwise MI between all questions. Score = sum of all MI values per question. |
| mi_mean | 89 | +0.242* | 0.022 | +0.328** | 0.002 | Pairwise MI between all questions. Score = mean of all MI values per question. |
| thresh_0.2_pagerank | 89 | +0.254* | 0.016 | +0.321** | 0.002 | Sparse network (|corr| > 0.2). PageRank centrality with |corr| weights. |
| complete_squared_harmonic | 89 | +0.225* | 0.034 | +0.318** | 0.002 | Complete network. Weight = corr². Harmonic centrality (sum of 1/distance). |
| complete_squared_closeness | 89 | +0.207 | 0.052 | +0.272** | 0.010 | Complete network. Weight = corr². Closeness centrality (1/avg path length). |
| complete_squared_strength | 89 | +0.205 | 0.055 | +0.272** | 0.010 | Complete network. Weight = corr². Strength = sum of edge weights per node. |
| complete_harmonic | 89 | +0.189 | 0.076 | +0.271* | 0.010 | Complete network. Weight = |corr|. Harmonic centrality. |
| complete_squared_pagerank | 89 | +0.201 | 0.059 | +0.263* | 0.013 | Complete network. Weight = corr². PageRank centrality. |
| complete_squared_katz | 89 | +0.205 | 0.054 | +0.257* | 0.015 | Complete network. Weight = corr². Katz centrality (paths attenuated by distance). |
| orig_Predictive | 89 | +0.186 | 0.081 | +0.253* | 0.017 | Linear model R² predicting all other questions from each question. |
| complete_log_harmonic | 89 | +0.173 | 0.104 | +0.248* | 0.019 | Complete network. Distance = -log(|corr|). Harmonic centrality. |
| complete_squared_eigenvector | 89 | +0.188 | 0.077 | +0.243* | 0.022 | Complete network. Weight = corr². Eigenvector centrality. |
| thresh_0.05_harmonic | 89 | +0.140 | 0.191 | +0.231* | 0.030 | Sparse network (|corr| > 0.05). Harmonic centrality. |
| orig_Network | 89 | +0.171 | 0.109 | +0.230* | 0.030 | Original composite: betweenness + closeness + eigenvector on corr network. |
| thresh_0.2_harmonic | 89 | +0.042 | 0.695 | +0.212* | 0.047 | Sparse network (|corr| > 0.2). Harmonic centrality. |
| thresh_0.2_strength | 89 | +0.174 | 0.102 | +0.210* | 0.048 | Sparse network (|corr| > 0.2). Strength = sum of remaining edge weights. |
| tree_max_degree_depth | 89 | -0.342** | 0.001 | -0.392*** | <.001 | Build Chow-Liu MST. Root = max degree. Raw depth (root=0). Negative: root→high CoG. |
| tree_max_neighbor_mi_depth | 89 | -0.365*** | <.001 | -0.420*** | <.001 | Build Chow-Liu MST. Root = max neighbor MI. Raw depth (root=0). Negative: root→high CoG. |

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
   - `betweenness`: Fraction of shortest paths passing through node
   - `closeness`: Inverse of average shortest path to all other nodes
   - `harmonic`: Sum of inverse distances (handles disconnected graphs)
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
