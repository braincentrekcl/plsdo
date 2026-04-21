# Interpreting PLS Output

## Cross-Correlation Heatmap

This matrix shows the Pearson correlation between every feature in X and
every feature in Y, computed across all subjects. It is the raw input to
the SVD. Strong positive or negative values indicate features that co-vary
across subjects.

## Singular Values and Permutation Test

The SVD breaks the cross-correlation matrix into latent variables (LVs),
ordered by how much covariance they explain. The singular value for each LV
quantifies its strength.

The permutation test asks: is this singular value larger than we would expect
if X and Y were unrelated? It shuffles the subject pairing between X and Y
10,000 times and compares the observed singular value to this null
distribution.

**How to read the plot:** A red line to the right of the grey histogram
indicates a singular value that exceeds the null distribution — that LV
captures real covariance, not noise.

## Loading Bar Plots

For each significant and reliable LV, the loading plots show which features
contribute most to the pattern. Bars are sorted by absolute loading. The
red error bars show the bootstrap standard error — they indicate how stable
each loading is across resampled versions of the data.

**Large bars with small error bars** are the features driving the pattern
reliably. **Large bars with large error bars** may be driven by a few
outlier subjects.

## Bootstrap Ratios

The bootstrap ratio is the loading divided by its standard error. It can be
interpreted like a z-score: values above 1.96 indicate that a feature's
contribution is reliable at the 95% confidence level.

## Subject Scores

Subject scores show how strongly each subject expresses a given LV pattern.
The X scores (XU) project each subject onto the X-side pattern; the Y
scores (YV') project onto the Y-side pattern.

**Box/strip plots** show how scores distribute across groups. If a LV
captures a group difference, the boxes will separate.

**Score scatter plots** (correlational PLS only) show the relationship
between X and Y scores. If the PLS pattern is strong, subjects should fall
along a diagonal. Group-specific linear fits reveal whether the X–Y
relationship differs by group.

## Cross-Validation (Discriminatory PLS)

Cross-validation tests whether the group discrimination holds on unseen
subjects. The fold accuracy histogram shows per-fold classification
accuracy, while the confusion matrix shows which groups are well-separated
and which are confused.

The permutation test of CV accuracy answers: is the observed accuracy
significantly better than chance? A p-value below 0.05 indicates that
the model generalises beyond the training data.

**Important:** do not select the number of components based on `plsdo run`
results and then feed that into cross-validation. This introduces
circularity. Use all components (the default) or use nested
cross-validation.
