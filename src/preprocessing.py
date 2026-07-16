"""Preprocessing helper functions for the bank account fraud detection project."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def temporal_split(
    df: pd.DataFrame,
    month_col: str = "month",
    target_col: str = "fraud_bool",
    train_months: Iterable[int] = range(0, 6),
    validation_month: int = 6,
    test_month: int = 7,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create the temporal train, validation, and test split used in the capstone."""
    train_mask = df[month_col].isin(list(train_months))
    valid_mask = df[month_col].eq(validation_month)
    test_mask = df[month_col].eq(test_month)

    X_train = df.loc[train_mask].drop(columns=[target_col])
    y_train = df.loc[train_mask, target_col].astype(int)
    X_valid = df.loc[valid_mask].drop(columns=[target_col])
    y_valid = df.loc[valid_mask, target_col].astype(int)
    X_test = df.loc[test_mask].drop(columns=[target_col])
    y_test = df.loc[test_mask, target_col].astype(int)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create conservative fraud-risk features used for exploration and modeling.

    The function is intentionally defensive: it creates features only when the
    required source columns exist.
    """
    output = df.copy()

    if "prev_address_months_count" in output.columns:
        output["has_unavailable_previous_address"] = (
            output["prev_address_months_count"].eq(-1).astype(int)
        )

    if "current_address_months_count" in output.columns:
        output["has_short_current_address_history"] = (
            output["current_address_months_count"].lt(12).astype(int)
        )

    if {"name_email_similarity", "phone_home_valid", "phone_mobile_valid"}.issubset(output.columns):
        output["identity_consistency_score"] = (
            output["name_email_similarity"].fillna(0)
            + output["phone_home_valid"].fillna(0)
            + output["phone_mobile_valid"].fillna(0)
        )

    if {"velocity_6h", "velocity_24h", "velocity_4w"}.issubset(output.columns):
        output["velocity_mean"] = output[["velocity_6h", "velocity_24h", "velocity_4w"]].mean(axis=1)

    return output


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the leakage-safe preprocessing transformer used by the models."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
