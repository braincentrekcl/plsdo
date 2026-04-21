# Missing Data in PLS

## Why Missing Data Is a Problem

PLS decomposes the cross-covariance between all features in X and all
features in Y simultaneously. Every subject must have a value for every
feature. A single missing value in one feature for one subject means that
subject cannot contribute to the covariance structure.

## What to Do

You have two choices, and which is better depends on your data:

### Drop Subjects

Remove subjects who are missing any measurement. This preserves all features
but reduces your sample size.

**When to choose this:** You have many subjects relative to features, and
only a few subjects have missing data.

### Drop Features

Remove features that have missing data across many subjects. This preserves
sample size but removes those features from the analysis.

**When to choose this:** A specific measurement failed for many subjects
(e.g. a brain region that was poorly imaged), but the remaining measurements
are complete.

### General Guidance

- Never impute missing data before PLS. Imputation introduces artificial
  covariance structure that PLS will happily decompose, producing patterns
  that reflect the imputation method rather than biology.
- Check the pattern of missingness before deciding. If it is random (a few
  scattered NaNs), dropping subjects is usually fine. If it is systematic
  (one feature missing for a whole group), that feature may be unreliable
  and should be dropped.
- The module intentionally does not make this decision for you. Prepare
  your input files so they contain only the subjects and features you want
  to analyse.
