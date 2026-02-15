# Table 1: Correlations between Polarization CoG and Fundamentalness Measures

*Best measures from exploratory analysis, sorted by Spearman ρ (N=88)*

| Measure | Pearson *r* | *p* | Spearman *ρ* | *p* |
|:--------|:-----------:|:---:|:------------:|:---:|
| MI: Maximum pairwise | +0.300** | 0.005 | +0.370*** | <.001 |
| MI: Composite (original) | +0.299** | 0.005 | +0.368*** | <.001 |
| Ensemble Score | +0.312** | 0.003 | +0.356*** | <.001 |
| Tree: Hierarchy centrality | +0.311** | 0.003 | +0.334** | 0.001 |
| MI: Average pairwise | +0.251* | 0.018 | +0.333** | 0.002 |
| Tree: Neighbor MI sum | +0.304** | 0.004 | +0.323** | 0.002 |
| MI: Average NMI | +0.242* | 0.023 | +0.317** | 0.003 |
| Tree: Composite | +0.337** | 0.001 | +0.316** | 0.003 |
| MI: Breadth (count significant) | +0.234* | 0.028 | +0.288** | 0.006 |
| Network: Harmonic | +0.204 | 0.056 | +0.277** | 0.009 |
| Network: Composite (original) | +0.185 | 0.085 | +0.230* | 0.031 |
| Tree: Depth (inverted) | -0.311** | 0.003 | -0.334** | 0.001 |

*\*p < .05, \*\*p < .01, \*\*\*p < .001*

## Notes
- Excluded: `polviews` (self-reported ideology), `partyid` (party identification)
- Dimensionality analysis removed from combined hierarchy (not significant, reduced N)
- Tree: Hierarchy centrality = 1 - normalized depth from root in Chow-Liu tree
- MI: Maximum = strongest single pairwise MI; Composite = weighted average of MI metrics
- Network: Harmonic centrality in correlation-weighted graph
- Tree: Depth inverted shows negative correlation (deeper = less fundamental)
