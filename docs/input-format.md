# Input Format

## Required Files

### X Matrix (correlational PLS only)

CSV with subjects as rows and features as columns. The first column must be
the subject identifier.

```csv
subject_id,region_A,region_B,region_C
sub01,1.23,4.56,7.89
sub02,2.34,5.67,8.90
```

### Y Matrix

Same format as X. Subject IDs must match across files (order does not matter
— the pipeline will align them).

### Demographics

CSV with a subject ID column and at least one grouping column.

```csv
subject_id,group,sex,age
sub01,control,F,25
sub02,treatment,M,30
```

## Optional Files

### Feature Metadata

CSV with a `feature` column matching data column headers, plus category
columns for plot colour-coding.

```csv
feature,category
region_A,frontal
region_B,frontal
region_C,temporal
```

### Groups Configuration

YAML file for multiple grouping variables. See `docs/usage.md` for examples.

## Subject Alignment

The pipeline finds the intersection of subject IDs across all input files.
Subjects present in some files but not others are excluded with a warning.
If no subjects are shared, the pipeline errors.

## Missing Data

The pipeline does **not** handle missing data. If any value in X or Y is
NaN, the pipeline errors and lists which subjects and features are affected.

See `docs/missing-data.md` for guidance on how to address this.
