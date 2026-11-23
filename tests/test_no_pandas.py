"""Tests for the pandas-free data layer (pandas DLLs are blocked by Windows Application Control)."""
import csv
import subprocess
import sys

import numpy as np
import pytest

from src import config as C
from src.data.load_data import COLUMNS, dataset_available, load_test, load_test_rul
from src.data.table import engine, from_records, groups, matrix, n_rows, sort_by, tail, take, to_records, unit_ids
from src.demo import simulate_engine

need_data = pytest.mark.skipif(not dataset_available(), reason="FD001 dataset not available")
need_model = pytest.mark.skipif(not (C.LSTM_CHECKPOINT.exists() and dataset_available()),
                                reason="trained model or dataset not available")


def small():
    return {"unit": np.array([2, 1, 2, 1]), "cycle": np.array([2, 2, 1, 1]), "x": np.array([4.0, 2.0, 3.0, 1.0])}


def test_table_helpers():
    t = small()
    assert n_rows(t) == 4 and unit_ids(t) == [1, 2]
    assert list(sort_by(t, "cycle")["cycle"]) == [1, 1, 2, 2]
    assert [(u, list(i)) for u, i in groups(t)] == [(1, [1, 3]), (2, [0, 2])]
    e = engine(t, 2)
    assert list(e["cycle"]) == [1, 2] and list(e["x"]) == [3.0, 4.0]
    assert n_rows(engine(t, 99)) == 0
    assert list(tail(t, 2)["x"]) == [3.0, 1.0] and n_rows(tail(t, 0)) == 0
    assert matrix(take(t, t["unit"] == 1), ["x", "cycle"]).tolist() == [[2.0, 2.0], [1.0, 1.0]]
    recs = to_records(t, ["cycle", "x"])
    assert recs[0] == {"cycle": 2, "x": 4.0} and type(recs[0]["cycle"]) is int
    assert list(from_records(recs)["x"]) == [4.0, 2.0, 3.0, 1.0]


def test_simulated_engine_is_a_table():
    t, info = simulate_engine(n_cycles=50, seed=1, stage="warning", profile=None)
    assert list(t) == COLUMNS and n_rows(t) == 50 and (t["unit"] == 999).all()


@need_data
def test_loader_shapes():
    test = load_test()
    assert list(test) == COLUMNS and test["unit"].dtype.kind == "i"
    assert unit_ids(test) == list(range(1, 101))
    assert len(load_test_rul()) == 100


def test_core_modules_do_not_import_pandas():
    code = ("import sys; sys.modules['pandas'] = None\n"  # any pandas import now raises ImportError
            "import src.train, src.evaluate, src.predict, src.demo, backend.main, pyarrow, plotly.graph_objects\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=C.ROOT)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


@need_model
def test_predictions_match_saved_pandas_era_results():
    """outputs/test_predictions.csv was written by the original pandas pipeline; the NumPy
    pipeline must reproduce those model predictions for every test engine."""
    from src.predict import Predictor
    with open(C.TEST_PREDICTIONS_PATH, newline="") as f:
        saved = {int(r["engine_id"]): float(r["predicted_rul"]) for r in csv.DictReader(f)}
    p, test = Predictor(), load_test()
    for eid, ref in saved.items():
        got = p.predict_last(engine(test, eid), mc_samples=0)["predicted_rul"]
        assert abs(got - ref) < 1e-2, (eid, got, ref)
