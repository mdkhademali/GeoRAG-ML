"""Model definitions, spatially-aware tuning and evaluation for GeoRAG-ML."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


def get_model_specs(seed: int = 42) -> Dict[str, Tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_grid)}. Grids are deliberately small (single-core friendly).

    Imputation (median) and scaling are inside the pipelines, so they are fitted on training folds only.
    Tree models do not need scaling; SVM and logistic regression do.
    """
    imp = lambda: ("impute", SimpleImputer(strategy="median"))  # noqa: E731
    return {
        "Logistic Regression": (
            Pipeline([imp(), ("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
            {"clf__C": [0.1, 1.0, 10.0]}),
        "SVM (RBF)": (
            Pipeline([imp(), ("scale", StandardScaler()),
                      ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed))]),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.05]}),
        "Random Forest": (
            Pipeline([imp(), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                            n_jobs=1, random_state=seed))]),
            {"clf__max_depth": [8, None], "clf__min_samples_leaf": [1, 5]}),
        "XGBoost": (
            Pipeline([imp(), ("clf", XGBClassifier(n_estimators=250, subsample=0.8, colsample_bytree=0.8,
                                                   eval_metric="logloss", n_jobs=1, random_state=seed,
                                                   tree_method="hist"))]),
            {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]}),
    }


def positive_scores(model, X) -> np.ndarray:
    """Continuous score for the positive class (probability if available, else decision function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve points."""
    pred = model.predict(X)
    score = positive_scores(model, X)
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]),
        "fpr": fpr, "tpr": tpr, "score": score, "pred": pred,
    }


def tune_and_evaluate(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
                      cv_splits: List[Tuple[np.ndarray, np.ndarray]], seed: int = 42,
                      model_names: List[str] | None = None) -> Tuple[Dict[str, dict], Dict[str, object]]:
    """Tune each model with buffered spatial-block CV on the training region, then score the held-out region.

    Returns (results, fitted_models). ``results[name]`` has best params, CV AUC (mean/std over folds for the
    selected configuration), held-out metrics and timing.
    """
    specs = get_model_specs(seed)
    names = model_names or list(specs)
    results, fitted = {}, {}
    for name in names:
        pipe, grid = specs[name]
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv_splits, refit=True, n_jobs=1)
        gs.fit(X_train, y_train)
        best_idx = gs.best_index_
        fold_scores = [gs.cv_results_[f"split{i}_test_score"][best_idx] for i in range(len(cv_splits))]
        ev = evaluate(gs.best_estimator_, X_test, y_test)
        ev.update({"best_params": gs.best_params_, "cv_auc_mean": float(np.mean(fold_scores)),
                   "cv_auc_std": float(np.std(fold_scores)), "fit_seconds": time.time() - t0})
        results[name], fitted[name] = ev, gs.best_estimator_
        log.info("%s | cv AUC %.3f | test AUC %.3f | %.1fs", name, ev["cv_auc_mean"], ev["roc_auc"], ev["fit_seconds"])
    return results, fitted


def results_table(results: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({"model": name, "accuracy": r["accuracy"], "precision": r["precision"],
                     "recall": r["recall"], "f1": r["f1"], "roc_auc": r["roc_auc"],
                     "cv_auc_mean": r["cv_auc_mean"], "cv_auc_std": r["cv_auc_std"]})
    return pd.DataFrame(rows).set_index("model").round(4)


def compare_validation_strategies(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray,
                                  spatial_splits: List[Tuple[np.ndarray, np.ndarray]], seed: int = 42) -> pd.DataFrame:
    """Same model, same data: random stratified K-fold vs buffered spatial-block K-fold (ROC-AUC).

    A large gap indicates optimism caused by spatial autocorrelation (neighbouring pixels in train and test).
    """
    n_splits = len(spatial_splits)
    rnd = cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                          scoring="roc_auc", n_jobs=1)
    spa = cross_val_score(pipe, X, y, cv=spatial_splits, scoring="roc_auc", n_jobs=1)
    return pd.DataFrame({"strategy": ["Random stratified K-fold", "Buffered spatial-block K-fold"],
                         "auc_mean": [rnd.mean(), spa.mean()], "auc_std": [rnd.std(), spa.std()],
                         "n_folds": [len(rnd), len(spa)]}).set_index("strategy").round(4)
