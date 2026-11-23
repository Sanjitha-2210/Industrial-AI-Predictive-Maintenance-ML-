"""pandas-free table rendering for the dashboard.

`st.dataframe` / `st.table` convert their input with pandas in a live Streamlit session (even for
pyarrow tables, via the lazy-loading path), and pandas' DLLs are blocked by Windows Application
Control on this machine. These helpers build an escaped HTML table from plain Python data instead.
"""
from __future__ import annotations

import html

import numpy as np


def format_cell(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def html_table(columns: dict, max_height: int | None = 420) -> str:
    """HTML table from {column name: sequence of values} (all sequences the same length)."""
    names = list(columns)
    cols = [list(columns[c]) for c in names]
    n = len(cols[0]) if cols else 0
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in names)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(format_cell(col[i]))}</td>" for col in cols) + "</tr>"
                   for i in range(n))
    style = f"max-height:{max_height}px;" if max_height else ""
    return (f"<div class='rul-table' style='overflow:auto;{style}'>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def records_to_columns(records: list[dict]) -> dict:
    """[{col: value}, ...] -> {col: [values]} (column order from the first record)."""
    if not records:
        return {}
    return {c: [r.get(c) for r in records] for c in records[0]}


TABLE_CSS = """
<style>
.rul-table table {border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums;}
.rul-table th, .rul-table td {padding:4px 8px; border-bottom:1px solid rgba(128,128,128,.25); text-align:right; white-space:nowrap;}
.rul-table th {position:sticky; top:0; background:var(--background-color, #fff); font-weight:600;}
.rul-table td:first-child, .rul-table th:first-child {text-align:left;}
</style>
"""
