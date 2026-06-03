import numpy as np
import pytest

from plsdo.stats import corrected_pvalue


class TestCorrectedPvalue:
    def test_observed_exceeds_all_null(self):
        """Nothing in the null beats the observed -> minimum p = 1/(n+1)."""
        null = np.arange(20.0)
        assert corrected_pvalue(100.0, null) == pytest.approx(1 / 21)

    def test_observed_below_all_null(self):
        """Everything in the null beats the observed -> p = 1.0."""
        null = np.arange(1.0, 21.0)
        assert corrected_pvalue(0.0, null) == pytest.approx(1.0)

    def test_correction_never_zero(self):
        null = np.zeros(10)
        assert corrected_pvalue(1e9, null) > 0.0

    def test_counts_ties_as_exceeding(self):
        """>= is inclusive: a null value equal to observed counts."""
        null = np.array([1.0, 2.0, 2.0, 3.0])
        # three values >= 2.0 (both 2s and the 3) -> (3 + 1) / (4 + 1)
        assert corrected_pvalue(2.0, null) == pytest.approx(4 / 5)

    def test_vectorised_matches_scalar(self):
        """Row-wise application (one null row per LV) matches scalar calls."""
        rng = np.random.default_rng(0)
        observed = np.array([2.0, 0.5, 1.0])
        null = rng.standard_normal((3, 50))
        vec = corrected_pvalue(observed, null, axis=1)
        for i in range(3):
            assert vec[i] == pytest.approx(corrected_pvalue(observed[i], null[i]))
