"""Reusable preprocessing utilities for the SubsurfaceBreaks experiments.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Constants for sorting the features
EUCLIDEAN_N = ["EuclideanNeighbor1_N", "EuclideanNeighbor2_N", "EuclideanNeighbor3_N"]
EUCLIDEAN_D = ["EuclideanNeighbor1_D", "EuclideanNeighbor2_D", "EuclideanNeighbor3_D"]
COSINE_N = ["CosineNeighbor1_N", "CosineNeighbor2_N", "CosineNeighbor3_N"]
COSINE_D = ["CosineNeighbor1_D", "CosineNeighbor2_D", "CosineNeighbor3_D"]
ANGLE_N = ["AngleNeighbor1_N", "AngleNeighbor2_N", "AngleNeighbor3_N"]
ANGLE_D = ["AngleNeighbor1_D", "AngleNeighbor2_D", "AngleNeighbor3_D"]

EUCLIDEAN_N_SORTED = ["Euclidean_N_Max", "Euclidean_N_Min", "Euclidean_N_Intermediate"]
EUCLIDEAN_D_SORTED = ["Euclidean_D_Max", "Euclidean_D_Min", "Euclidean_D_Intermediate"]
COSINE_N_SORTED = ["Cosine_N_Max", "Cosine_N_Min", "Cosine_N_Intermediate"]
COSINE_D_SORTED = ["Cosine_D_Max", "Cosine_D_Min", "Cosine_D_Intermediate"]
ANGLE_N_SORTED = ["Angle_N_Max", "Angle_N_Min", "Angle_N_Intermediate"]
ANGLE_D_SORTED = ["Angle_D_Max", "Angle_D_Min", "Angle_D_Intermediate"]

SORTING_PAIRS = [
    (EUCLIDEAN_N, EUCLIDEAN_N_SORTED),
    (EUCLIDEAN_D, EUCLIDEAN_D_SORTED),
    (COSINE_N, COSINE_N_SORTED),
    (COSINE_D, COSINE_D_SORTED),
    (ANGLE_N, ANGLE_N_SORTED),
    (ANGLE_D, ANGLE_D_SORTED),
]

# Helpers
SYNTHETIC_PASSTHROUGH_COLS = ["X_N", "Y_N", "Z_N", "X_D", "Y_D", "Z_D"]
# X_C, Y_C, Z_C are the coordinates of the center of the triangle
REAL_PASSTHROUGH_COLS = ["X_C", "Y_C", "Z_C", "X_N", "Y_N", "Z_N", "X_D", "Y_D", "Z_D"]


def load_raw_synthetic(input_path: str, n_files: int = 1000) -> pd.DataFrame:
    """Concatenate the raw, unfiltered synthetic triangle files into a single DataFrame.
    As in the original paper Michalak et al. (2025).

    The method is important for the replication purposes (random-split with the seed).
    """
    dfs = [
        pd.read_csv(
            os.path.join(input_path, f"{file_number}.txt"), decimal=".", sep=";"
        )
        for file_number in range(n_files)
    ]
    return pd.concat(dfs, ignore_index=True)


def filter_triangles(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the quality filter (as in the original paper Michalak et al. (2025)).
    """
    mask = (
        (df["X_C_Neighbor1"] != "undefined")
        & (df["X_C_Neighbor2"] != "undefined")
        & (df["X_C_Neighbor3"] != "undefined")
        & (df["Z_N"] != 0)
        & (df["n1_zn"] != 0)
        & (df["n2_zn"] != 0)
        & (df["n3_zn"] != 0)
        & (df["DOC"] < 0.90)
    )
    return df[mask].reset_index(drop=True)


def sort_values(row: pd.Series, output_columns: list) -> pd.Series:
    """Sort a row's Neighbor values into (max, min, intermediate).

    Parameters
    ----------
    row : pd.Series
        A row containing the three Neighbor values for one distance group (e.g. Euclidean_N).
    output_columns : list
        Output column names: [max_col, min_col, intermediate_col].

    Returns
    -------
    pd.Series
        max value, min value, remaining ("intermediate") value.
    """
    max_val = row.max()
    min_val = row.min()
    remaining_val = row.sum() - max_val - min_val
    return pd.Series([max_val, min_val, remaining_val], index=output_columns)


def build_sorted_features(
    df: pd.DataFrame,
    sorting_pairs: Iterable[tuple],
    sort_values_fn,
    passthrough_cols: Sequence[str],
) -> pd.DataFrame:
    """Build the Max/Min/Intermediate sorted feature frame plus passthrough columns.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered input frame (synthetic `filtered_df` or real `real_filtered_df`).
    sorting_pairs : Iterable[tuple]
        `(raw_cols, sorted_cols)` pairs, e.g. `SORTING_PAIRS` constant.
    sort_values_fn : callable
        Row-wise sorter, e.g. `sort_values` function.
    passthrough_cols : Sequence[str]
        Raw columns kept as-is at the front of the output (e.g.
        `SYNTHETIC_PASSTHROUGH_COLS` or `REAL_PASSTHROUGH_COLS` constants).
    """
    sorted_parts = [
        df[cols].apply(sort_values_fn, axis=1, output_columns=sorted_cols)
        for cols, sorted_cols in sorting_pairs
    ]
    return pd.concat([df[passthrough_cols], *sorted_parts], axis=1)


def load_and_filter_real(input_path: str, real_file: str) -> pd.DataFrame:
    """Load the real dataset and apply the same quality filter as for the synthetic dataset.
    """
    real_raw_df = pd.read_csv(os.path.join(input_path, real_file), decimal=".", sep=";")
    return filter_triangles(real_raw_df)


def method2_scale_per_group(
    df: pd.DataFrame, feature_cols: Sequence[str], group_col: str = "File_number"
) -> pd.DataFrame:
    """Method 2: an independent `StandardScaler` fit per group (per simulation/file).
    """
    return df.groupby(group_col)[feature_cols].transform(lambda s: (s - s.mean()) / s.std(ddof=0))


def grouped_cv_fault_counts_by_scaling_method(
    X: pd.DataFrame | None,
    y: pd.Series,
    groups: pd.Series,
    method: Literal["method2", "method3"],
    model: Literal["logistic_regression", "random_forest"],
    z_precomputed: pd.DataFrame | None = None,
    n_splits: int = 10,
    positive_class: int = 1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Grouped K-fold check of observed vs expected fault counts, methods 2) and 3).

    Every group (simulations/files) is predicted by a model that never saw it
    (leave-one-group-out cross-validation).

    Parameters
    ----------
    X : pd.DataFrame
        Raw (unscaled) feature columns. Ignored when `method == "method2"`.
    y : pd.Series
        Target column.
    groups : pd.Series
        Group id per row (e.g. `File_number`).
    method : {"method2", "method3"}
        Which scaling method to apply inside each fold. On synthetic data, method 1)
        reduces to method 3) (one common scale).
    model : {"logistic_regression", "random_forest"}
        Which classifier to use.
    z_precomputed : pd.DataFrame, optional
        The pre-scaled feature frame (same row order as `X`/`y`/`groups`).
        Required when `method == "method2"`.
    n_splits : int
        Number of `GroupKFold` folds.
    positive_class : int
        The positive class (fault).
    random_state : int
        Passed to the classifier for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        One row per group: `group_id`, `n`, `observed_faults`, `expected_faults`,
        `count_error` (difference between expected and observed fault counts);
        and the out-of-group probability of the positive class.
    """
    if method not in {"method2", "method3"}:
        raise ValueError(f"Unknown method: {method}. Expected 'method2' or 'method3'.")
    if model not in {"logistic_regression", "random_forest"}:
        raise ValueError(
            f"Unknown model: {model}. "
            "Expected 'logistic_regression' or 'random_forest'."
        )
    if method == "method2" and z_precomputed is None:
        raise ValueError("Method 2 requires z_precomputed parameter. Got None.")

    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    X_array = None if method == "method2" else np.asarray(X)
    Z_array = np.asarray(z_precomputed) if method == "method2" else None

    probabilities = np.full(len(y_array), np.nan)
    group_kfold = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in group_kfold.split(
        np.zeros(len(y_array)), y_array, groups_array
    ):
        if method == "method2":
            Z_train, Z_test = Z_array[train_idx], Z_array[test_idx]
        elif method == "method3":  # Elif to avoid misleading "else"
            scaler_train = StandardScaler().fit(X_array[train_idx])
            Z_train = scaler_train.transform(X_array[train_idx])
            Z_test = scaler_train.transform(X_array[test_idx])

        if model == "logistic_regression":
            classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model == "random_forest":
            # min_samples_leaf - 10% of the training sample size; Malley et al. (2012)
            classifier = RandomForestClassifier(
                min_samples_leaf=int(0.1*len(y_array[train_idx]))+1, n_jobs=-1, random_state=random_state
            )
        
        classifier.fit(Z_train, y_array[train_idx])
        positive_class_index = int(
            np.flatnonzero(classifier.classes_ == positive_class)[0]
        )
        
        # DEBUG prints.
        print(f"Positive class index: {positive_class_index}")
        print(f"Classifier classes: {classifier.classes_}")
        print(f"Selected class: {classifier.classes_[positive_class_index]}")

        probabilities[test_idx] = classifier.predict_proba(Z_test)[
            :, positive_class_index
        ]

    per_row = pd.DataFrame(
        {
            "group_id": groups_array,
            "is_fault": (y_array == positive_class).astype(int),
            "probability": probabilities,
        }
    )
    per_group = (
        per_row.groupby("group_id")
        .agg(
            n=("is_fault", "size"),
            observed_faults=("is_fault", "sum"),
            expected_faults=("probability", "sum"),
        )
        .reset_index()
    )
    per_group["count_error"] = (
        per_group["expected_faults"] - per_group["observed_faults"]
    )
    return per_group, probabilities
