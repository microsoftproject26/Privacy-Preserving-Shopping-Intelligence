"""Explicit feature whitelist and train-only sklearn pipelines for T2."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = (
    "category_code",
    "log_query_price",
    "log_prior_events",
    "log_prior_views",
    "log_prior_carts",
    "log_prior_purchases",
    "log_prior_distinct_items",
    "log_same_item_prior_events",
    "log_session_elapsed_seconds",
    "log_previous_gap_seconds",
    "hour_sin",
    "hour_cos",
    "weekend",
)


def feature_matrix(columns: dict[str, Any]) -> np.ndarray:
    """Accept exactly predictor columns; identities/outcomes never reach the estimator."""
    if set(columns) != set(FEATURES):
        raise ValueError("feature columns must match the explicit predictor whitelist")
    arrays = [np.asarray(columns[name]) for name in FEATURES]
    if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("features must be aligned 1D columns")
    x = np.column_stack([np.asarray(columns[name], dtype=np.float64) for name in FEATURES])
    if x.ndim != 2 or len(x) == 0 or np.isinf(x).any():
        raise ValueError("features must be a nonempty aligned matrix with no infinity")
    cat = x[:, 0]
    if (
        not np.isfinite(cat).all()
        or not np.equal(cat, np.floor(cat)).all()
        or ((cat < -1) | (cat >= 588)).any()
    ):
        raise ValueError("category feature is a dense code 0..587 or upstream unknown -1")
    return x


def make_pipeline(
    model: str, params: dict[str, Any], *, seed: int = 13, threads: int = 2
) -> Pipeline:
    expected = {"C"} if model == "logistic_regression" else {"num_leaves", "reg_lambda"}
    if set(params) != expected:
        raise ValueError("parameters do not match the preregistered model family")
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
                [0],
            ),
            ("numeric", numeric, list(range(1, len(FEATURES)))),
        ],
        sparse_threshold=1.0,
    )
    if model == "logistic_regression":
        estimator = LogisticRegression(
            C=float(params["C"]),
            solver="lbfgs",
            max_iter=500,
            tol=1e-4,
            class_weight=None,
            random_state=seed,
        )
    elif model == "lightgbm":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=int(params["num_leaves"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_samples=100,
            max_bin=63,
            subsample=1.0,
            colsample_bytree=1.0,
            class_weight=None,
            random_state=seed,
            n_jobs=threads,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
    else:
        raise ValueError(f"unknown baseline: {model}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def positive_probabilities(pipeline: Pipeline, x: np.ndarray) -> np.ndarray:
    classes = np.asarray(pipeline.named_steps["model"].classes_)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("T2 training requires both binary classes")
    p = np.asarray(pipeline.predict_proba(x)[:, 1], dtype=np.float64)
    if p.shape != (len(x),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid purchase probabilities")
    return p
