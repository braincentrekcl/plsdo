# Statistical Methods

This page documents methodological choices in `plsdo` that differ from
or extend common PLS implementations.
For a general introduction to interpreting the output, see
[Interpreting output](interpreting-output.md).

## Design matrix encoding (discriminatory PLS)

Discriminatory PLS constructs a design matrix from categorical group
labels.
`plsdo` uses **full-rank dummy coding**: a factor with *k* levels
produces *k* indicator columns (one per level), not *k − 1*.

The rows of this matrix sum to 1, so the design has rank *k − 1* and
the SVD returns *k − 1* non-trivial latent variables.
Group contrasts are implicitly referenced to the grand mean.

This follows the convention in [@mcintosh2004] and differs from
packages that use mean-centred deviation coding (e.g. `plscmd` in
Matlab), where the reference level is explicit.
The resulting singular values and loadings are mathematically
equivalent; the difference is interpretive, not computational.

## Permutation p-values

Permutation p-values use the corrected formula from [@phipson2010]:

    p = (c + 1) / (n_perms + 1)

where *c* is the count of permuted singular values ≥ the observed
value.
This guarantees exact control of the Type I error rate at any number
of permutations and avoids p-values of exactly zero.

The same correction is applied to the cross-validation permutation
test.

## Latent variable filtering

`plsdo` applies a two-stage filter before reporting latent variables
as significant:

1. The permutation test p-value must be < 0.05.
2. At least one feature must have |bootstrap ratio| > 1.96 on
   **both** the X and Y sides.

Most PLS implementations check only the Y side.
Requiring reliability on both sides is a conservative choice that
reduces false positives where a significant singular value is driven
by noise on the X side.
The threshold of 1.96 corresponds to a 95% confidence interval under
the standard-normal approximation to the bootstrap distribution.

## Bootstrap alignment

Each bootstrap resample produces a new SVD, whose singular vectors
may be rotated or sign-flipped relative to the reference.
`plsdo` aligns bootstrap replicates using Procrustes rotation on the
Y-side loadings (Vt), then applies a sign correction based on the
dot product with the reference loadings.

Aligning in a single space (Vt) is sufficient because the SVD
guarantees a shared coordinate system and the rotation matrix Q is
orthogonal.
This follows the approach in [@mcintosh2004].

## References

BibTeX entries are in [references.bib](references.bib).

- **mcintosh2004** — McIntosh AR, Lobaugh NJ (2004).
  Partial least squares analysis of neuroimaging data: applications
  and advances. *NeuroImage*, 23, S250–S263.
  doi:[10.1016/j.neuroimage.2004.07.020](https://doi.org/10.1016/j.neuroimage.2004.07.020)
- **krishnan2011** — Krishnan A, Williams LJ, McIntosh AR, Abdi H (2011).
  Partial Least Squares (PLS) methods for neuroimaging: a tutorial
  and review. *NeuroImage*, 56(2), 455–475.
  doi:[10.1016/j.neuroimage.2010.07.034](https://doi.org/10.1016/j.neuroimage.2010.07.034)
- **mcintosh2013** — McIntosh AR, Mišić B (2013).
  Multivariate statistical analyses for neuroimaging data.
  *Annual Review of Psychology*, 64, 499–525.
  doi:[10.1146/annurev-psych-113011-143804](https://doi.org/10.1146/annurev-psych-113011-143804)
- **phipson2010** — Phipson B, Smyth GK (2010).
  Permutation P-values should never be zero: calculating exact
  P-values when permutations are randomly drawn.
  *Stat Appl Genet Mol Biol*, 9(1).
  doi:[10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585)
