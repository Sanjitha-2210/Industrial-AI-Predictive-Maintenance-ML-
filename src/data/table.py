"""Minimal column table used instead of pandas: a dict {column name: 1-D NumPy array}.

pandas is avoided because its compiled parser DLL is blocked by Windows Application Control
on the development machine. All columns in a table have the same length.
"""
from __future__ import annotations

import numpy as np

Table = dict[str, np.ndarray]


def n_rows(t: Table) -> int:
    return len(next(iter(t.values()))) if t else 0


def take(t: Table, idx) -> Table:
    """Rows selected by an index array or boolean mask."""
    return {k: v[idx] for k, v in t.items()}


def tail(t: Table, n: int) -> Table:
    return {k: v[-n:] if n > 0 else v[:0] for k, v in t.items()}


def sort_by(t: Table, col: str) -> Table:
    return take(t, np.argsort(t[col], kind="stable"))


def unit_ids(t: Table) -> list[int]:
    return [int(u) for u in np.unique(t["unit"])]


def groups(t: Table):
    """Yield (unit, row indices) in ascending unit order; rows keep their original order."""
    for u in np.unique(t["unit"]):
        yield int(u), np.flatnonzero(t["unit"] == u)


def engine(t: Table, unit: int) -> Table:
    """All rows of one engine sorted by cycle (empty table if the unit is absent)."""
    return sort_by(take(t, t["unit"] == unit), "cycle")


def matrix(t: Table, cols) -> np.ndarray:
    """(rows, len(cols)) float matrix."""
    return np.column_stack([np.asarray(t[c], dtype=float) for c in cols]) if len(cols) else np.empty((n_rows(t), 0))


def from_records(records: list[dict], columns=None) -> Table:
    columns = columns or list(records[0].keys())
    return {c: np.array([r[c] for r in records]) for c in columns}


def to_records(t: Table, columns=None) -> list[dict]:
    """JSON-friendly rows (native Python int/float)."""
    columns = columns or list(t.keys())
    cols = {c: t[c].tolist() for c in columns}
    return [{c: cols[c][i] for c in columns} for i in range(n_rows(t))]
