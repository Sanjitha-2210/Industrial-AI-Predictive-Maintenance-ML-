"""CSV Dataset Validation and Model Compatibility Checker.

Validates external time-series sensor datasets without pandas.
Detects columns, data types, missing values, duplicates, time-series continuity,
machine-level record counts, sequence lengths, and model safety constraints.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAX_UPLOAD_BYTES = 2048 * 1024 * 1024  # 2 GB limit for large industrial CSV/TXT
MAX_ZIP_UPLOAD_BYTES = 2048 * 1024 * 1024  # 2 GB limit for large ZIP archives
SAFE_EXTENSIONS = {".csv", ".txt", ".zip", ".tsv", ".dat", ".mat"}



@dataclass
class ColumnProfile:
    name: str
    dtype: str  # "float", "int", "string", "empty"
    total_count: int
    null_count: int
    unique_count: int
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetValidationReport:
    status: str  # "VALID" | "WARNING" | "REJECTED"
    filename: str
    row_count: int
    column_count: int
    columns: list[str]
    column_profiles: dict[str, ColumnProfile]
    machine_id_candidates: list[str]
    cycle_candidates: list[str]
    rul_candidates: list[str]
    sensor_candidates: list[str]
    selected_machine_col: str | None = None
    selected_cycle_col: str | None = None
    selected_rul_col: str | None = None
    machine_count: int = 0
    machine_ids: list[Any] = field(default_factory=list)
    min_sequence_length: int = 0
    max_sequence_length: int = 0
    avg_sequence_length: float = 0.0
    duplicate_record_count: int = 0
    has_monotonic_cycles: bool = True
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    nasa_compatible: bool = False
    nasa_compatibility_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        # Ensure ColumnProfile objects are serializable
        res["column_profiles"] = {k: asdict(v) for k, v in self.column_profiles.items()}
        return res


def _try_float(val: str) -> tuple[bool, float | None]:
    val = val.strip()
    if val == "" or val.lower() in ("nan", "null", "none", "na", "n/a"):
        return True, None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return True, None
        return True, f
    except ValueError:
        return False, None


def read_csv_data(filepath_or_bytes: str | Path | bytes, max_rows: int | None = None) -> tuple[list[str], list[list[str]]]:
    """Safely parse CSV into headers and string rows without pandas."""
    if isinstance(filepath_or_bytes, bytes):
        text = filepath_or_bytes.decode("utf-8", errors="replace")
        f = io.StringIO(text)
    else:
        path = Path(filepath_or_bytes)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        f = open(path, "r", encoding="utf-8", errors="replace")

    try:
        reader = csv.reader(f)
        header = None
        rows = []
        for i, row in enumerate(reader):
            if not row or all(c.strip() == "" for c in row):
                continue
            if header is None:
                header = [c.strip() for c in row]
            else:
                rows.append([c.strip() for c in row])
                if max_rows and len(rows) >= max_rows:
                    break
        if header is None:
            raise ValueError("CSV file is empty or missing headers.")
        return header, rows
    finally:
        f.close()


def validate_csv_dataset(
    filepath_or_bytes: str | Path | bytes,
    filename: str = "dataset.csv",
    machine_col: str | None = None,
    cycle_col: str | None = None,
    rul_col: str | None = None,
    required_seq_len: int = 30,
) -> DatasetValidationReport:
    """Performs deep validation of an external sensor dataset."""
    issues: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    # File size check if path given
    if isinstance(filepath_or_bytes, (str, Path)):
        p = Path(filepath_or_bytes)
        if p.suffix.lower() not in SAFE_EXTENSIONS:
            return DatasetValidationReport(
                status="REJECTED",
                filename=filename,
                row_count=0,
                column_count=0,
                columns=[],
                column_profiles={},
                machine_id_candidates=[],
                cycle_candidates=[],
                rul_candidates=[],
                sensor_candidates=[],
                issues=[f"Unsupported file extension '{p.suffix}'. Only .csv and .txt are supported."],
            )
        if p.stat().st_size > MAX_UPLOAD_BYTES:
            return DatasetValidationReport(
                status="REJECTED",
                filename=filename,
                row_count=0,
                column_count=0,
                columns=[],
                column_profiles={},
                machine_id_candidates=[],
                cycle_candidates=[],
                rul_candidates=[],
                sensor_candidates=[],
                issues=[f"File size exceeds maximum allowed upload size of {MAX_UPLOAD_BYTES // (1024*1024)} MB."],
            )

    try:
        columns, raw_rows = read_csv_data(filepath_or_bytes)
    except Exception as e:
        return DatasetValidationReport(
            status="REJECTED",
            filename=filename,
            row_count=0,
            column_count=0,
            columns=[],
            column_profiles={},
            machine_id_candidates=[],
            cycle_candidates=[],
            rul_candidates=[],
            sensor_candidates=[],
            issues=[f"CSV parsing error: {str(e)}"],
        )

    n_rows = len(raw_rows)
    n_cols = len(columns)

    if n_rows == 0:
        return DatasetValidationReport(
            status="REJECTED",
            filename=filename,
            row_count=0,
            column_count=n_cols,
            columns=columns,
            column_profiles={},
            machine_id_candidates=[],
            cycle_candidates=[],
            rul_candidates=[],
            sensor_candidates=[],
            issues=["CSV contains header but zero data rows."],
        )

    # Analyze column profiles
    col_profiles: dict[str, ColumnProfile] = {}
    col_values_float: dict[str, list[float | None]] = {c: [] for c in columns}
    col_is_numeric: dict[str, bool] = {c: True for c in columns}

    for row_idx, row in enumerate(raw_rows):
        if len(row) != n_cols:
            issues.append(f"Row {row_idx + 1} has {len(row)} values, expected {n_cols}.")
            if len(issues) > 10:
                issues.append("Additional malformed rows truncated.")
                break

    for col_idx, col_name in enumerate(columns):
        raw_vals = [row[col_idx] if col_idx < len(row) else "" for row in raw_rows]
        is_num = True
        numeric_vals: list[float] = []
        null_count = 0
        samples = []

        for v in raw_vals:
            is_valid_num, fval = _try_float(v)
            if not is_valid_num:
                is_num = False
            if fval is None:
                null_count += 1
                col_values_float[col_name].append(None)
            else:
                numeric_vals.append(fval)
                col_values_float[col_name].append(fval)
            if len(samples) < 5 and v != "":
                samples.append(v)

        col_is_numeric[col_name] = is_num and len(numeric_vals) > 0
        unique_c = len(set(raw_vals))

        if col_is_numeric[col_name] and numeric_vals:
            arr = np.array(numeric_vals, dtype=float)
            dtype = "int" if np.all(arr == np.floor(arr)) else "float"
            col_profiles[col_name] = ColumnProfile(
                name=col_name,
                dtype=dtype,
                total_count=n_rows,
                null_count=null_count,
                unique_count=unique_c,
                min_value=float(np.min(arr)),
                max_value=float(np.max(arr)),
                mean_value=float(np.mean(arr)),
                sample_values=samples,
            )
        else:
            col_profiles[col_name] = ColumnProfile(
                name=col_name,
                dtype="string",
                total_count=n_rows,
                null_count=null_count,
                unique_count=unique_c,
                sample_values=samples,
            )

    # Detect candidate roles
    machine_candidates: list[str] = []
    cycle_candidates: list[str] = []
    rul_candidates: list[str] = []
    sensor_candidates: list[str] = []

    lower_to_orig = {c.lower(): c for c in columns}

    for col in columns:
        low = col.lower()
        if any(term in low for term in ("unit", "engine", "machine", "asset", "device", "pump", "turbine", "motor", "id")):
            machine_candidates.append(col)
        elif any(term in low for term in ("cycle", "time", "step", "timestamp", "hour", "day", "epoch")):
            cycle_candidates.append(col)
        elif any(term in low for term in ("rul", "remaining", "life", "ttf", "time_to_failure")):
            rul_candidates.append(col)
        elif col_is_numeric[col]:
            sensor_candidates.append(col)

    # Resolve roles
    m_col = machine_col or (machine_candidates[0] if machine_candidates else None)
    c_col = cycle_col or (cycle_candidates[0] if cycle_candidates else None)
    r_col = rul_col or (rul_candidates[0] if rul_candidates else None)

    # Check null values in sensors
    high_null_cols = [c for c, p in col_profiles.items() if p.null_count > 0.05 * n_rows]
    if high_null_cols:
        warnings.append(f"Columns with >5% missing values detected: {', '.join(high_null_cols)}.")
        recommendations.append("Impute or drop columns with high missing rates prior to model training.")

    # Multi-machine analysis
    machine_ids = []
    seq_lens = []
    duplicate_count = 0
    monotonic = True

    if m_col and m_col in columns:
        raw_m_vals = [row[columns.index(m_col)] for row in raw_rows]
        # unique preserved order
        seen = set()
        for v in raw_m_vals:
            if v not in seen:
                seen.add(v)
                machine_ids.append(v)

        # per-machine sequence length and time order
        machine_to_cycles: dict[str, list[float]] = {mid: [] for mid in machine_ids}
        c_idx = columns.index(c_col) if (c_col and c_col in columns) else None

        for row in raw_rows:
            mid = row[columns.index(m_col)]
            if c_idx is not None:
                _, fval = _try_float(row[c_idx])
                if fval is not None:
                    machine_to_cycles[mid].append(fval)

        for mid, cycles in machine_to_cycles.items():
            seq_lens.append(len(cycles) if c_idx is not None else raw_m_vals.count(mid))
            if cycles:
                # check duplicates
                if len(set(cycles)) != len(cycles):
                    duplicate_count += len(cycles) - len(set(cycles))
                # check monotonicity
                if any(cycles[k] >= cycles[k + 1] for k in range(len(cycles) - 1)):
                    monotonic = False
    else:
        # Single machine assumed
        machine_ids = ["machine_1"]
        seq_lens = [n_rows]
        if c_col and c_col in columns:
            c_vals = [col_values_float[c_col][i] for i in range(n_rows) if col_values_float[c_col][i] is not None]
            if len(set(c_vals)) != len(c_vals):
                duplicate_count = len(c_vals) - len(set(c_vals))
            if any(c_vals[k] >= c_vals[k + 1] for k in range(len(c_vals) - 1)):
                monotonic = False

    min_seq = int(min(seq_lens)) if seq_lens else 0
    max_seq = int(max(seq_lens)) if seq_lens else 0
    avg_seq = float(np.mean(seq_lens)) if seq_lens else 0.0

    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate cycle/timestamp records within machine histories.")
        recommendations.append("Deduplicate observations before creating sliding temporal windows.")

    if not monotonic:
        warnings.append("Time/cycle indices are not strictly monotonic ascending.")
        recommendations.append("Sort dataset by (machine_id, cycle) to preserve causal time series order.")

    if min_seq < required_seq_len:
        warnings.append(
            f"Minimum machine sequence length ({min_seq}) is shorter than recommended window length ({required_seq_len}). "
            f"Short sequences will require zero/front-padding."
        )

    # NASA model compatibility check
    nasa_features = [
        "op_setting_1", "op_setting_2", "sensor_2", "sensor_3", "sensor_4",
        "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
        "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    ]
    missing_nasa = [f for f in nasa_features if f not in columns]
    if len(missing_nasa) == 0:
        nasa_compatible = True
        nasa_compatibility_reason = (
            "Dataset matches NASA C-MAPSS 17-feature schema exactly. "
            "Compatible with NASA C-MAPSS baseline model."
        )
    else:
        nasa_compatible = False
        nasa_compatibility_reason = (
            f"Dataset is NOT directly compatible with the NASA C-MAPSS model. "
            f"Missing required NASA sensor features: {missing_nasa[:5]}... "
            f"Use Mode 3 (Industry Adaptation) to map sensors and adapt or train a model for this equipment."
        )

    # Overall Status determination
    if len(issues) > 0:
        status = "REJECTED"
    elif len(warnings) > 0:
        status = "WARNING"
    else:
        status = "VALID"

    return DatasetValidationReport(
        status=status,
        filename=filename,
        row_count=n_rows,
        column_count=n_cols,
        columns=columns,
        column_profiles=col_profiles,
        machine_id_candidates=machine_candidates,
        cycle_candidates=cycle_candidates,
        rul_candidates=rul_candidates,
        sensor_candidates=sensor_candidates,
        selected_machine_col=m_col,
        selected_cycle_col=c_col,
        selected_rul_col=r_col,
        machine_count=len(machine_ids),
        machine_ids=machine_ids[:200],  # cap list length for response size
        min_sequence_length=min_seq,
        max_sequence_length=max_seq,
        avg_sequence_length=round(avg_seq, 2),
        duplicate_record_count=duplicate_count,
        has_monotonic_cycles=monotonic,
        issues=issues,
        warnings=warnings,
        recommendations=recommendations,
        nasa_compatible=nasa_compatible,
        nasa_compatibility_reason=nasa_compatibility_reason,
    )
