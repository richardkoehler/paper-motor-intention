"""Helper functions for statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
from statannotations.stats.StatTest import StatTest


def wilcoxon() -> StatTest:
    """Wrapper for StatTest with scipy Wilcoxon signed-rank test."""

    def _stat_test(
        x: pd.Series | np.ndarray, y: pd.Series | np.ndarray
    ):  # type: ignore
        diff = x - y
        return scipy.stats.wilcoxon(
            diff,
            y=None,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="auto",
        )

    return StatTest(
        func=_stat_test,
        alpha=0.05,
        test_long_name="Wilcoxon Signed-Rank",
        test_short_name="Wilcoxon",
    )


def mannwhitneyu() -> StatTest:
    """Wrapper for StatTest with scipy Wilcoxon signed-rank test."""

    def _stat_test(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
        return scipy.stats.mannwhitneyu(
            x,
            y=y,
            alternative="two-sided",
        )

    return StatTest(
        func=_stat_test,
        alpha=0.05,
        test_long_name="Mann-Whitney U",
        test_short_name="MWU",
    )


def permutation_onesample() -> StatTest:
    """Wrapper for StatTest with permutation one-sample test."""

    def _stat_test(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
        # if isinstance(x, pd.Series):
        #     x = x.to_numpy()
        # if isinstance(y, pd.Series):
        #     y = y.to_numpy()
        # diff = x - y
        # return pte_stats.permutation_onesample(
        #     data_a=diff, data_b=0, n_perm=100000, two_tailed=True
        # )
        res = scipy.stats.permutation_test(
            (x - y,),
            np.mean,
            vectorized=True,
            n_resamples=int(1e6),
            permutation_type="samples",
        )
        return res.statistic, res.pvalue

    return StatTest(
        func=_stat_test,
        alpha=0.05,
        test_long_name="Permutation Test",
        test_short_name="Perm. Test",
    )


def permutation_twosample() -> StatTest:
    """Wrapper for StatTest with permutation two-sample test."""

    def statistic(x, y, axis):
        return np.mean(a=x, axis=axis) - np.mean(a=y, axis=axis)

    def _stat_test(
        x: pd.Series | np.ndarray, y: pd.Series | np.ndarray
    ) -> tuple[float, float]:
        # if isinstance(x, pd.Series):
        #     x = x.to_numpy()
        # if isinstance(y, pd.Series):
        #     y = y.to_numpy()
        # return pte_stats.permutation_twosample(
        #     data_a=x, data_b=y, n_perm=100000, two_tailed=True
        # )
        res = scipy.stats.permutation_test(
            (x, y),
            statistic,
            vectorized=True,
            n_resamples=int(1e6),
            permutation_type="independent",
        )
        return res.statistic, res.pvalue

    return StatTest(
        func=_stat_test,
        alpha=0.05,
        test_long_name="Permutation Test",
        test_short_name="Perm. Test",
    )
