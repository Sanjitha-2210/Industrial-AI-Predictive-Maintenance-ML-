"""Dashboard table rendering without pandas (st.dataframe/st.table use pandas in a live session)."""
import re
import subprocess
import sys

import numpy as np

from src import config as C
from src.data.table import tail

sys.path.insert(0, str(C.ROOT / "dashboard"))
from table_view import format_cell, html_table, records_to_columns  # noqa: E402

APP = C.ROOT / "dashboard" / "app.py"


def test_html_table_structure_and_escaping():
    h = html_table({"cycle": [1, 2], "name": ["<b>x</b>", "y&z"]})
    assert h.count("<tr>") == 3 and "<th>cycle</th><th>name</th>" in h
    assert "&lt;b&gt;x&lt;/b&gt;" in h and "y&amp;z" in h and "<b>x</b>" not in h


def test_format_cell():
    assert format_cell(np.int64(31)) == "31" and format_cell(np.float64(518.67)) == "518.67"
    assert format_cell(-0.0006) == "-0.0006" and format_cell(True) == "True" and format_cell("rising") == "rising"


def test_sensor_history_last_15_rows_without_unit():
    n = 40
    df = {"unit": np.full(n, 7), "cycle": np.arange(1, n + 1), "sensor_2": np.linspace(640, 645, n)}
    h = html_table({c: v.tolist() for c, v in tail(df, 15).items() if c != "unit"})
    assert "<th>unit</th>" not in h and h.count("<tr>") == 16
    assert re.search(r"<tr><td>26</td>", h) and re.search(r"<tr><td>40</td>", h) and "<td>25</td>" not in h


def test_records_to_columns():
    recs = [{"sensor": "sensor_11", "drift_std_units": 1.5, "direction": "rising"},
            {"sensor": "sensor_12", "drift_std_units": -1.2, "direction": "falling"}]
    assert records_to_columns(recs) == {"sensor": ["sensor_11", "sensor_12"], "drift_std_units": [1.5, -1.2],
                                        "direction": ["rising", "falling"]}
    assert records_to_columns([]) == {}


def test_app_uses_no_pandas_backed_streamlit_tables():
    for f in (APP, APP.parent / "components.py", APP.parent / "theme.py"):
        src = f.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        for banned in ("st.dataframe(", "st.table(", "st.data_editor(", "import pandas", "import pyarrow"):
            assert banned not in code.split('"""', 2)[-1], (f.name, banned)


def test_table_view_imports_without_pandas():
    code = ("import sys; sys.modules['pandas'] = None; sys.path.insert(0, 'dashboard')\n"
            "import table_view; print(table_view.html_table({'a': [1]})[:5])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=C.ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
