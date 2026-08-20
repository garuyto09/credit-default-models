"""Model definitions for the default prediction benchmark.

Every estimator is wrapped in a Pipeline so that scaling is fitted inside each
cross-validation fold rather than on the full dataset. This matters more than it
looks: KNN and Logistic Regression are distance/penalty based, and the raw features
span wildly different ranges (LIMIT_BAL reaches 1,000,000 NT$ while AGE sits in the
tens). Without scaling, the KNN neighbourhood is decided almost entirely by the
credit limit column.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def _scaled(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_models() -> dict[str, Pipeline]:
    """The four candidate models, in increasing order of flexibility.

    Logistic Regression is the reference: it is what most deployed scorecards still
    are, because it is monotonic, auditable and easy to translate into points. The
    tree ensembles are here to measure how much signal a linear model leaves behind.
    """
    return {
        "Logistic Regression": _scaled(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "KNN": _scaled(KNeighborsClassifier(n_neighbors=11)),
        "Decision Tree": _scaled(
            DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
        ),
        "Random Forest": _scaled(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }
