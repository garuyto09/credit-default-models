"""Discrimination, calibration and stability metrics for credit scoring models.

Accuracy is close to useless on a credit portfolio: at a 22% default rate, a model
that predicts "nobody defaults" is already 78% accurate. The metrics that actually
drive decisions are the ones below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: maximum separation between the cumulative distributions
    of defaulters and non-defaulters.

    The industry's default discrimination metric. Read as: how far apart can the
    model pull the two populations? Below ~0.20 the model is rarely worth deploying;
    0.30-0.40 is a healthy behavioural scorecard.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1.

    Same information as AUC, rescaled so that 0 is a coin flip and 1 is perfect
    ranking. Reported instead of AUC in most credit risk contexts.
    """
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """Population Stability Index between a reference and a new score distribution.

    PSI answers a different question from KS: not "does the model separate?" but
    "is the population it sees today still the population it was built on?". It is
    the metric that catches silent model decay in production.

    Conventional reading:
        PSI < 0.10   population stable
        0.10 - 0.25  moderate shift, investigate
        PSI > 0.25   significant shift, model likely needs redevelopment
    """
    # Quantile cuts from the reference distribution, so each expected bin holds ~10%.
    breakpoints = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Guard against empty bins, which would send the log term to infinity.
    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def decile_table(y_true: np.ndarray, y_score: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """Risk band table: the artefact a credit committee actually reads.

    Sorts the portfolio from riskiest to safest, cuts it into bands, and shows the
    observed default rate in each. A model is useful when this table is monotonic
    and the top band is several times riskier than the bottom one — that ratio is
    what a cut-off policy is built on.
    """
    df = pd.DataFrame({"y": y_true, "score": y_score})
    # rank-then-cut avoids ties collapsing bands when scores are discrete
    df["band"] = pd.qcut(df["score"].rank(method="first"), n_bands, labels=False)
    df["band"] = n_bands - df["band"]  # band 1 = highest predicted risk

    table = (
        df.groupby("band")
        .agg(accounts=("y", "size"), defaults=("y", "sum"), avg_score=("score", "mean"))
        .reset_index()
        .sort_values("band")
    )
    table["default_rate"] = table["defaults"] / table["accounts"]
    table["lift"] = table["default_rate"] / df["y"].mean()
    table["cum_defaults_pct"] = table["defaults"].cumsum() / table["defaults"].sum()
    table["cum_accounts_pct"] = table["accounts"].cumsum() / table["accounts"].sum()

    return table


def summarize(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Headline metrics for one model."""
    return {
        "ks": ks_statistic(y_true, y_score),
        "gini": gini(y_true, y_score),
        "auc": float(roc_auc_score(y_true, y_score)),
    }
