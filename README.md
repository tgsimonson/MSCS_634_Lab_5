# MSCS 634 Lab 5: Hierarchical and DBSCAN Clustering

**Todd Simonson**
MSCS-634: Advanced Big Data and Data Mining
Dr. Satish Penmatsa

## Purpose

This lab applies two clustering algorithms to the Wine dataset from scikit-learn (178 wines, 3 cultivars, 13 chemical measurements): **Agglomerative Hierarchical Clustering** and **DBSCAN**. The features were standardized first. The notebook then examines how each algorithm's parameters shape the result:

- **Hierarchical clustering:** `n_clusters` and the linkage method
- **DBSCAN:** `eps` and `min_samples`

Clusters are shown as 2-D PCA scatter plots and a dendrogram. Results are scored with the Silhouette, Homogeneity, and Completeness scores, plus the Adjusted Rand Index (ARI). The cultivar labels were used only to score the results, never to form the clusters.

## Repository Contents

| File | Description |
|---|---|
| `MSCS_634_Lab_5.ipynb` | Notebook with data exploration, standardization, both algorithms, parameter experiments, visualizations, metrics, and analysis |
| `requirements.txt` | Python packages needed to run the notebook |

The Wine dataset ships with scikit-learn (`sklearn.datasets.load_wine`), so no data file is needed.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook MSCS_634_Lab_5.ipynb
```

## Results

| | Hierarchical (Ward, 3 clusters) | DBSCAN (eps 2.4, min_samples 5) |
|---|---|---|
| Clusters found | 3 | 2 |
| Noise points | 0 | 36 (20%) |
| Silhouette | 0.277 | 0.327 (non-noise points) / 0.196 (all points) |
| Homogeneity | **0.790** | 0.504 |
| Completeness | **0.783** | 0.552 |
| ARI | **0.790** | 0.421 |

## Key Insights

- **Hierarchical clustering with Ward linkage recovered the three cultivars.** 165 of 178 wines landed in a cluster matching their class. Every error was a class_1 wine.
- **Three clusters is the natural number.** For Ward linkage, silhouette and ARI both peak at n_clusters = 3. With more clusters, homogeneity rises (0.790 to 0.835) while completeness falls (0.783 to 0.553), the expected trade-off. The dendrogram confirms 3 without using the labels: the merge distance jumps from 12.6 to 27.7 at the last two merges, and class_2 joins last, which makes it the most distinct cultivar.
- **Linkage mattered more than the number of clusters.** Ward and complete linkage found meaningful groups. Average and single linkage chained through the data into one 174-wine cluster plus two tiny ones (ARI near 0).
- **DBSCAN found two clusters, not three.** It separated class_2 almost perfectly (42 of 43 wines), merged class_0 with the dense part of class_1, and labeled 36 wines as noise, 78% of them class_1.
- **DBSCAN's higher silhouette is misleading.** It is computed only on the 142 wines DBSCAN kept and ignores the 36 hardest cases. Including them drops DBSCAN to 0.196, below Ward's 0.277.
- **Both algorithms agree on the data's shape.** Class_2 is the most distinct cultivar, and class_1 is a spread-out group that touches both others.
- **Dimensionality hurts DBSCAN.** Run on the 2-D PCA projection, it finds the three cultivar groups (ARI 0.663 versus 0.421 in 13 dimensions).

## Challenges and Decisions

1. **The usual eps heuristic failed.** The elbow of the k-distance plot (min_samples = 5) is at eps = 2.71, but every eps from 2.6 upward merges almost all wines into one cluster. eps = 2.4 was selected instead, using unsupervised criteria only: it is the largest eps, and so produces the least noise, before the clusters merge, and the two-cluster result is stable across nearby settings. The class labels were not used to choose the parameters.
2. **A narrow working range for DBSCAN.** A 50-setting grid (10 eps values × 5 min_samples values) showed that useful results exist only between about eps = 2.2 and 2.4 at min_samples = 5. A change of 0.2 turns two clusters into mostly noise or into one cluster. Heatmaps of cluster count, noise, and silhouette across the grid make this sensitivity visible.
3. **Scoring noise points.** Silhouette for DBSCAN is reported on non-noise points, which is standard practice, and also with noise counted as its own group, so the comparison with hierarchical clustering is fair. Homogeneity and completeness treat noise as its own label, scikit-learn's default, and are also shown for non-noise points alone.
4. **Plotting 13 dimensions.** All clustering ran on the full 13 standardized features. The scatter plots use the first two principal components, which capture 55.4% of the variance, so some points that look close in a plot are farther apart in the full space.
5. **Testing the dimensionality explanation.** To check whether DBSCAN's weak result came from the high dimensionality, it was rerun on the 2-D PCA projection as a supplementary experiment. The improvement there supports that explanation.
