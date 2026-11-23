"""Dataset Profiling Engine.

Computes distributions, outliers, and inter-sensor correlations using pure NumPy
(pandas-free architecture).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.adaptation.validator import read_csv_data


@dataclass
class SensorDistribution:
    name: str
    count: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    skewness: float
    outlier_count_3sigma: int
    outlier_percentage: float


@dataclass
class DatasetProfile:
    filename: str
    row_count: int
    machine_count: int
    machine_id_column: str | None
    sensor_distributions: list[SensorDistribution]
    correlation_matrix: dict[str, dict[str, float]]
    sensor_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["sensor_distributions"] = [asdict(d) for d in self.sensor_distributions]
        return res


def compute_dataset_profile(
    filepath_or_bytes: str | Path | bytes,
    sensor_columns: list[str],
    machine_col: str | None = None,
) -> DatasetProfile:
    """Profiles sensor distributions and correlation matrix."""
    header, raw_rows = read_csv_data(filepath_or_bytes)
    n_rows = len(raw_rows)
    if n_rows == 0:
        raise ValueError("Cannot profile an empty dataset.")

    # Validate sensors exist in header
    valid_sensors = [s for s in sensor_columns if s in header]
    if not valid_sensors:
        raise ValueError("None of the specified sensor columns exist in the dataset.")

    col_indices = [header.index(s) for s in valid_sensors]

    # Convert sensor columns to float matrix
    matrix_data = []
    valid_row_mask = []
    for r in raw_rows:
        try:
            row_vals = [float(r[idx]) for idx in col_indices]
            matrix_data.append(row_vals)
            valid_row_mask.append(True)
        except (ValueError, IndexError):
            valid_row_mask.append(False)

    if not matrix_data:
        raise ValueError("No valid numeric rows found for the selected sensor columns.")

    X = np.array(matrix_data, dtype=float)  # (N, F)

    # Calculate distributions
    dists: list[SensorDistribution] = []
    for i, sname in enumerate(valid_sensors):
        col = X[:, i]
        mean_val = float(np.mean(col))
        std_val = float(np.std(col))
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        q25 = float(np.percentile(col, 25))
        median_val = float(np.median(col))
        q75 = float(np.percentile(col, 75))

        # Skewness: m3 / s^3
        diff = col - mean_val
        m3 = np.mean(diff ** 3)
        skew = float(m3 / (std_val ** 3)) if std_val > 1e-9 else 0.0

        # Outliers > 3 sigma
        if std_val > 1e-9:
            outliers = np.sum(np.abs(diff) > 3.0 * std_val)
        else:
            outliers = 0
        outlier_pct = round((outliers / len(col)) * 100.0, 2)

        dists.append(
            SensorDistribution(
                name=sname,
                count=len(col),
                mean=round(mean_val, 4),
                std=round(std_val, 4),
                min=round(min_val, 4),
                q25=round(q25, 4),
                median=round(median_val, 4),
                q75=round(q75, 4),
                max=round(max_val, 4),
                skewness=round(skew, 4),
                outlier_count_3sigma=int(outliers),
                outlier_percentage=outlier_pct,
            )
        )

    # Compute correlation matrix
    corr_matrix: dict[str, dict[str, float]] = {}
    if len(valid_sensors) > 1 and X.shape[0] > 1:
        # standard correlation with safeguard against 0 variance
        stds = np.std(X, axis=0)
        non_zero_var = stds > 1e-9
        R = np.eye(len(valid_sensors))
        if np.all(non_zero_var):
            R = np.corrcoef(X, rowvar=False)

        for i, s1 in enumerate(valid_sensors):
            corr_matrix[s1] = {}
            for j, s2 in enumerate(valid_sensors):
                val = float(R[i, j]) if not np.isnan(R[i, j]) else 0.0
                corr_matrix[s1][s2] = round(val, 4)
    else:
        for s in valid_sensors:
            corr_matrix[s] = {s: 1.0}

    # Count machines
    machine_count = 1
    if machine_col and machine_col in header:
        m_idx = header.index(machine_col)
        m_vals = set(r[m_idx] for i, r in enumerate(raw_rows) if valid_row_mask[i])
        machine_count = len(m_vals)

    fname = str(filepath_or_bytes) if isinstance(filepath_or_bytes, (str, Path)) else "dataset.csv"

    return DatasetProfile(
        filename=Path(fname).name,
        row_count=len(matrix_data),
        machine_count=machine_count,
        machine_id_column=machine_col,
        sensor_distributions=dists,
        correlation_matrix=corr_matrix,
        sensor_names=valid_sensors,
    )
