"""Run the full benchmark: train the models, score them, write figures and results.

    python run.py

Outputs land in reports/ — four figures plus a results.md table that the README
links to. Everything is seeded, so a rerun reproduces the committed numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from src import data, metrics, models

REPORTS = Path(__file__).resolve().parent / "reports"
PALETTE = {
    "Logistic Regression": "#2563eb",
    "KNN": "#f59e0b",
    "Decision Tree": "#16a34a",
    "Random Forest": "#dc2626",
}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)

    X, y = data.load()
    print(f"Loaded {len(X):,} accounts · {X.shape[1]} features · "
          f"default rate {y.mean():.2%}")

    # Stratified so the 22% default rate is preserved in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=models.RANDOM_STATE
    )

    scores: dict[str, np.ndarray] = {}
    rows = []

    for name, model in models.build_models().items():
        print(f"Fitting {name} ...")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        scores[name] = proba

        summary = metrics.summarize(y_test.to_numpy(), proba)
        # PSI between the train and test score distributions. On a random split it
        # should be near zero — it is here as a baseline for what "stable" looks
        # like before the same check is run against a later origination cohort.
        train_proba = model.predict_proba(X_train)[:, 1]
        summary["psi_train_vs_test"] = metrics.population_stability_index(
            train_proba, proba
        )
        summary["model"] = name
        rows.append(summary)

    results = pd.DataFrame(rows)[
        ["model", "ks", "gini", "auc", "psi_train_vs_test"]
    ].sort_values("ks", ascending=False)

    print("\n" + results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    _plot_roc(y_test, scores)
    _plot_ks(y_test, scores)
    _plot_calibration(y_test, scores)
    _plot_lift(y_test, scores, results.iloc[0]["model"])
    _write_results(results, y_test, scores)

    print(f"\nFigures and results written to {REPORTS}/")


def _plot_roc(y_test, scores) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, proba in scores.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        g = metrics.gini(y_test.to_numpy(), proba)
        ax.plot(fpr, tpr, color=PALETTE[name], lw=2, label=f"{name} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "roc.png", dpi=150)
    plt.close(fig)


def _plot_ks(y_test, scores) -> None:
    """KS is the vertical gap between the two cumulative curves — draw it literally."""
    fig, axes = plt.subplots(1, len(scores), figsize=(4 * len(scores), 3.6), sharey=True)
    for ax, (name, proba) in zip(axes, scores.items()):
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        ks_idx = np.argmax(tpr - fpr)
        ax.plot(thresholds, tpr, color="#dc2626", lw=1.8, label="Defaulters")
        ax.plot(thresholds, fpr, color="#2563eb", lw=1.8, label="Non-defaulters")
        ax.vlines(
            thresholds[ks_idx], fpr[ks_idx], tpr[ks_idx],
            color="#111827", lw=1.4, linestyle="--",
        )
        ax.set_title(f"{name}\nKS = {tpr[ks_idx] - fpr[ks_idx]:.3f}", fontsize=10)
        ax.set_xlabel("Score threshold")
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cumulative share")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "ks.png", dpi=150)
    plt.close(fig)


def _plot_calibration(y_test, scores) -> None:
    """Ranking well is not the same as being right about the level.

    A scorecard used for pricing or expected-loss provisioning needs the predicted
    probability itself to be trustworthy, not just its ordering.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, proba in scores.items():
        true_p, pred_p = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(pred_p, true_p, "o-", color=PALETTE[name], lw=1.8, ms=4, label=name)
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="Perfect calibration")
    ax.set_xlabel("Predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=150)
    plt.close(fig)


def _plot_lift(y_test, scores, best_model: str) -> None:
    table = metrics.decile_table(y_test.to_numpy(), scores[best_model])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["band"], table["default_rate"], color="#2563eb", width=0.7)
    ax.axhline(y_test.mean(), color="#dc2626", lw=1.4, linestyle="--",
               label=f"Portfolio average ({y_test.mean():.1%})")
    ax.set_xlabel("Risk band (1 = riskiest decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Default rate by risk band — {best_model}")
    ax.set_xticks(table["band"])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "risk_bands.png", dpi=150)
    plt.close(fig)


def _write_results(results, y_test, scores) -> None:
    best = results.iloc[0]["model"]
    table = metrics.decile_table(y_test.to_numpy(), scores[best])

    lines = [
        "# Results",
        "",
        "Generated by `run.py`. All figures live alongside this file.",
        "",
        "## Discrimination",
        "",
        results.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Risk bands — {best}",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (REPORTS / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
