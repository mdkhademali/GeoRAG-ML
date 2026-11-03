"""SHAP-based explanation helpers for tree models (Random Forest, XGBoost).

TreeSHAP gives exact Shapley values for tree ensembles (Lundberg et al., 2020). SHAP values describe how the
*fitted model* uses its inputs; they are not causal effects on the real-world process.
"""
from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def transform_before_classifier(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every pipeline step except the final classifier (e.g. the median imputer)."""
    Xt = X
    for name, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    return pd.DataFrame(np.asarray(Xt), columns=X.columns, index=X.index)


def compute_shap_values(pipe: Pipeline, X_explain: pd.DataFrame, background: pd.DataFrame | None = None,
                        max_samples: int = 1500, seed: int = 42) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Return (shap_values for the positive class, the explained feature frame, expected value).

    Handles the different output shapes of shap for sklearn forests ((n, f, 2)) and XGBoost ((n, f)).
    """
    if len(X_explain) > max_samples:
        X_explain = X_explain.sample(max_samples, random_state=seed)
    Xt = transform_before_classifier(pipe, X_explain)
    clf = pipe.steps[-1][1]
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):                      # older shap: list of per-class arrays
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                              # (n, f, classes)
        sv = sv[:, :, 1]
    ev = explainer.expected_value
    ev = float(np.ravel(ev)[-1])
    return sv, Xt, ev


def shap_importance(sv: np.ndarray, feature_names: Sequence[str]) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted descending."""
    return pd.Series(np.abs(sv).mean(axis=0), index=list(feature_names)).sort_values(ascending=False)


def permutation_importance_table(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10,
                                 seed: int = 42) -> pd.DataFrame:
    """Model-agnostic permutation importance (drop in ROC-AUC) on data NOT used for fitting."""
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "auc_drop_mean": r.importances_mean, "auc_drop_std": r.importances_std})
            .sort_values("auc_drop_mean", ascending=False).reset_index(drop=True))
