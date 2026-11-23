import numpy as np
import pytest
import torch

from src.data.load_data import COLUMNS, DatasetMissingError, load_train
from src.data.preprocess import Preprocessor, add_rul, make_windows, split_engines, windows_for_engine
from src.data.table import matrix, take
from src.models.datf_net import DATFNet
from src.models.lstm import LSTMRegressor


def fake_df(n_units=6, seed=0):
    rng = np.random.default_rng(seed)
    lengths = [40 + 5 * u for u in range(1, n_units + 1)]
    N = sum(lengths)
    t = {c: rng.normal(size=N) for c in COLUMNS[2:]}
    t["sensor_1"] = np.full(N, 5.0)  # constant channel must be dropped
    t["unit"] = np.repeat(np.arange(1, n_units + 1), lengths)
    t["cycle"] = np.concatenate([np.arange(1, T + 1) for T in lengths])
    return t


def test_rul_labels_and_clip():
    df = add_rul(fake_df(), clip=20)
    g = take(df, df["unit"] == 1)
    assert g["RUL"][-1] == 0 and g["RUL"].max() == 20


def test_engine_split_has_no_overlap():
    tr, va = split_engines(range(1, 101), 0.2, seed=1)
    assert not set(tr) & set(va) and len(va) == 20 and len(tr) == 80


def test_scaler_fit_on_train_only_and_constant_dropped():
    df = fake_df()
    tr = take(df, df["unit"] <= 4)
    prep = Preprocessor(10, 20).fit(tr, [5, 6])
    assert "sensor_1" not in prep.feature_cols
    scaled = prep.transform(tr)
    assert np.abs(matrix(scaled, prep.feature_cols).mean(axis=0)).max() < 1e-6  # zero mean on train data


def test_windows_shape_and_labels():
    df = add_rul(fake_df(), clip=100)
    prep = Preprocessor(10, 100).fit(df)
    X, y, u = make_windows(prep.transform(df), prep.feature_cols, 10)
    assert X.shape[1:] == (10, len(prep.feature_cols)) and len(X) == len(y) == len(u)
    assert (y[u == 1] == df["RUL"][df["unit"] == 1][9:]).all()


def test_short_engine_padding():
    w, ends = windows_for_engine(np.arange(5, dtype=float).reshape(5, 1), 8)
    assert w.shape == (5, 8, 1) and (w[0, :, 0] == 0).all() and w[-1, -1, 0] == 4


def test_legacy_band_function_only_reproduces_frozen_evaluation_column():
    """src.predict.health_status is kept only for the frozen src/evaluate.py; it must still reproduce the historical
    CSV column exactly, and nothing at runtime may use it."""
    import csv
    from pathlib import Path

    from src import config as C
    from src.predict import health_status
    if C.TEST_PREDICTIONS_PATH.exists():
        with open(C.TEST_PREDICTIONS_PATH, newline="") as f:
            for r in csv.DictReader(f):
                assert health_status(float(r["predicted_rul"])) == r["health_status"]
    for f in ("backend/main.py", "dashboard/app.py", "dashboard/components.py", "src/health_state.py"):
        assert "health_status(" not in (Path(C.ROOT) / f).read_text(encoding="utf-8"), f


def test_model_forward_shapes():
    x = torch.randn(4, 30, 17)
    assert LSTMRegressor(17)(x).shape == (4,)
    out = DATFNet()(x, torch.randn(4, 17))
    assert out["rul"].shape == (4,) and out["hi"].shape == (4, 30)


def test_missing_dataset_message(tmp_path):
    with pytest.raises(DatasetMissingError, match="Download"):
        load_train(tmp_path)
