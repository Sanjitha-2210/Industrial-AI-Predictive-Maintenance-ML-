"""DATF-Net model, degradation-aware data pipeline, and experiment-protocol guards."""
import json

import numpy as np
import pytest
import torch

from src import config as C
from src.data.degradation import BASELINE_CYCLES, build_windows, engine_baseline
from src.data.load_data import dataset_available
from src.models.datf_net import (DATFConfig, DATFNet, count_parameters, hi_supervision_loss, monotonicity_loss,
                                 slope_weights)

need_data = pytest.mark.skipif(not (dataset_available() and C.PREPROCESSOR_PATH.exists()),
                               reason="dataset or saved preprocessor not available")
VARIANTS = [{}, {"use_degradation": False}, {"use_baseline": False}, {"use_slope": False}, {"pooling": "last"},
            {"fusion": "concat"}, {"encoder": "lstm"}]


# ---------------- model ----------------
@pytest.mark.parametrize("overrides", VARIANTS)
def test_forward_shapes_all_variants(overrides):
    m = DATFNet(DATFConfig(**overrides)).eval()
    x, b = torch.randn(4, 30, 17), torch.randn(4, 17)
    out = m(x, b)
    assert out["rul"].shape == (4,)
    if overrides.get("use_degradation", True):
        assert out["hi"].shape == (4, 30) and ((out["hi"] > 0) & (out["hi"] < 1)).all()
    else:
        assert out["hi"] is None


def test_parameter_budget_matches_spec():
    assert count_parameters(DATFNet()) == 54_626  # specification; LSTM baseline has 60,801


def test_tcn_is_causal():
    m = DATFNet().eval()
    x = torch.randn(2, 30, 17)
    H = m.encode(x)
    x2 = x.clone()
    x2[:, 20:] += 5.0  # change the future
    assert torch.allclose(m.encode(x2)[:, :20], H[:, :20], atol=1e-6)


def test_baseline_is_used_and_required():
    m = DATFNet().eval()
    x = torch.randn(2, 30, 17)
    with pytest.raises(ValueError):
        m(x, None)
    assert not torch.allclose(m(x, torch.zeros(2, 17))["rul"], m(x, torch.ones(2, 17))["rul"])
    m2 = DATFNet(DATFConfig(use_baseline=False)).eval()
    assert torch.allclose(m2(x, torch.zeros(2, 17))["rul"], m2(x, torch.ones(2, 17))["rul"])


def test_slope_filter_recovers_linear_trend():
    w = slope_weights(30)
    t = torch.arange(30, dtype=torch.float32)
    assert abs(float((w * (3.0 + 0.25 * t)).sum()) - 0.25) < 1e-5 and abs(float(w.sum())) < 1e-6


def test_hi_losses_direction():
    decreasing = torch.linspace(1, 0, 30)[None].repeat(3, 1)
    assert monotonicity_loss(decreasing) == 0
    assert monotonicity_loss(decreasing.flip(1)) > 0  # health increasing over time is penalised
    assert hi_supervision_loss(decreasing, decreasing) == 0 and hi_supervision_loss(decreasing, 1 - decreasing) > 0


def test_invalid_config():
    for bad in ({"encoder": "gru"}, {"pooling": "max"}, {"fusion": "sum"}):
        with pytest.raises(ValueError):
            DATFNet(DATFConfig(**bad))


# ---------------- data pipeline ----------------
def synthetic_table(lengths=(40, 55), clip=125):
    rng = np.random.default_rng(0)
    rows = {"unit": [], "cycle": [], "f1": [], "f2": [], "RUL": []}
    for u, n in enumerate(lengths, start=1):
        rows["unit"] += [u] * n
        rows["cycle"] += list(range(1, n + 1))
        rows["f1"] += list(rng.normal(size=n))
        rows["f2"] += list(np.arange(n) / 10.0)
        rows["RUL"] += list(np.minimum(np.arange(n)[::-1], clip))
    return {k: np.array(v) for k, v in rows.items()}


def test_baseline_uses_only_first_k_cycles():
    F = np.arange(40 * 2, dtype=float).reshape(40, 2)
    assert np.allclose(engine_baseline(F, 10), F[:10].mean(0))
    F2 = F.copy()
    F2[10:] += 100  # later cycles must not affect the baseline
    assert np.allclose(engine_baseline(F2, 10), engine_baseline(F, 10))
    assert np.allclose(engine_baseline(F[:4], 10), F[:4].mean(0))  # short history: use what exists


def test_windows_targets_and_hi_definition():
    t = synthetic_table(clip=30)
    d = build_windows(t, ["f1", "f2"], 10, 30, units=[1, 2], k=5)
    n1 = 40 - 10 + 1
    assert d["X"].shape == (n1 + 55 - 10 + 1, 10, 2) and d["hi_target"].shape == (len(d["X"]), 10)
    # y = RUL at the last window cycle; HI target = min(RUL_t, clip)/clip for each cycle of the window
    assert d["y"][0] == min(40 - 10, 30) and d["y"][n1 - 1] == 0
    assert np.allclose(d["hi_target"][:, -1], d["y"] / 30)
    assert (d["hi_target"] >= 0).all() and (d["hi_target"] <= 1).all()
    assert (np.diff(d["hi_target"], axis=1) <= 0).all()          # direction: non-increasing health
    assert d["hi_target"][n1 - 1, -1] == 0 and d["hi_target"][0, 0] == 1  # failure -> 0, healthy (RUL >= clip) -> 1
    assert np.allclose(d["b"][0], np.array([t["f1"][:5].mean(), t["f2"][:5].mean()]))
    assert (d["end_cycle"] >= 5).all()                            # baseline cycles precede every window end


@need_data
def test_same_windows_split_and_scaler_as_baseline():
    from src.data.degradation import load_train_val
    from src.data.load_data import load_train
    from src.data.preprocess import Preprocessor, add_rul, make_windows
    prep = Preprocessor.load()
    train, val = load_train_val(prep)
    scaled = prep.transform(add_rul(load_train(), prep.rul_clip))
    all_units = set(int(u) for u in np.unique(scaled["unit"]))
    Xt, yt, _ = make_windows(scaled, prep.feature_cols, prep.seq_len, all_units - set(prep.val_units))
    Xv, yv, _ = make_windows(scaled, prep.feature_cols, prep.seq_len, set(prep.val_units))
    assert np.array_equal(train["X"], Xt) and np.array_equal(train["y"], yt)
    assert np.array_equal(val["X"], Xv) and np.array_equal(val["y"], yv)
    assert set(np.unique(val["unit"])) == set(prep.val_units) and not set(np.unique(train["unit"])) & set(prep.val_units)
    assert train["X"].shape == (14459, 30, 17) and val["X"].shape == (3272, 30, 17)


@need_data
def test_test_split_matches_baseline_last_windows():
    from src.data.degradation import load_test_split
    from src.data.load_data import load_test
    from src.data.preprocess import Preprocessor, last_windows
    prep = Preprocessor.load()
    t = load_test_split(prep)
    X, ids = last_windows(prep.transform(load_test()), prep.feature_cols, prep.seq_len)
    assert np.array_equal(t["X"], X) and np.array_equal(t["engine_id"], ids) and len(t["true_rul"]) == 100


# ---------------- training / protocol ----------------
def tiny_data(n=64, T=30, F=17, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 125, n).astype(np.float32)
    hi = np.minimum(y[:, None] + np.arange(T)[::-1][None, :], 125).astype(np.float32) / 125
    return {"X": rng.normal(size=(n, T, F)).astype(np.float32), "y": y, "b": rng.normal(size=(n, F)).astype(np.float32),
            "hi_target": hi, "unit": np.repeat(np.arange(8), n // 8)}


def test_training_is_deterministic_and_records_run(tmp_path):
    from src.train_datf import RunSpec, TrainSettings, train_run
    data = tiny_data()
    settings = TrainSettings(epochs=2, lambda_hi=0.3, lambda_mono=0.1)
    spec = RunSpec("t", "datf", {}, 0.3, 0.1)
    h1 = train_run(spec, data, data, 7, settings, 125.0, tmp_path / "a.pt", torch.device("cpu"), log=lambda s: None)
    h2 = train_run(spec, data, data, 7, settings, 125.0, tmp_path / "b.pt", torch.device("cpu"), log=lambda s: None)
    assert h1["train_loss"] == h2["train_loss"] and h1["val_loss"] == h2["val_loss"]
    assert h1["train_total_loss"][0] > h1["train_loss"][0]  # auxiliary HI losses are part of the objective
    ck = torch.load(tmp_path / "a.pt", weights_only=False)
    assert ck["seed"] == 7 and ck["settings"]["lambda_hi"] == 0.3 and ck["datf_config"]["encoder"] == "tcn"


@need_data
def test_tuning_stage_never_touches_test_data(tmp_path, monkeypatch):
    """The tuning / selection stage must run with every test-data loader disabled."""
    import src.data.degradation as deg
    import src.data.load_data as ld
    import src.experiments as ex
    from src.train_datf import TrainSettings

    def boom(*a, **k):
        raise AssertionError("test data accessed during tuning")
    for mod, name in ((deg, "load_test"), (deg, "load_test_rul"), (deg, "load_test_split"), (ld, "load_test"),
                      (ld, "load_test_rul"), (ex, "load_test_split")):
        monkeypatch.setattr(mod, name, boom)
    ex.run_stage("tune", ex.tune_specs()[:2], [0], TrainSettings(epochs=1), exp_dir=tmp_path / "exp",
                 ckpt_dir=tmp_path / "ck", evaluate_test=False, log=lambda s: None)
    monkeypatch.setattr(ex, "TUNE_GRID", ex.TUNE_GRID[:2])
    sel = ex.select_lambdas(exp_dir=tmp_path / "exp", seeds=[0])
    assert sel["selected"]["run"] in ("datf_lh0.1_lm0.0", "datf_lh0.1_lm0.1")
    for f in (tmp_path / "exp").rglob("run.json"):
        assert "test" not in json.loads(f.read_text())


def test_bootstrap_detects_known_difference():
    from src.experiments import bootstrap_rmse_diff
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 150, 100)
    good = true + rng.normal(0, 5, (3, 100))
    bad = true + rng.normal(0, 15, (3, 100))
    r = bootstrap_rmse_diff(good, bad, true, n=2000)
    assert r["rmse_diff"] < 0 and r["ci95"][1] < 0


@need_data
def test_final_stage_with_reused_checkpoint_writes_records(tmp_path):
    """Regression: a reused tuning checkpoint must still get its run folder, run.json and test predictions."""
    import src.experiments as ex
    from src.train_datf import TrainSettings
    ex.run_stage("tune", ex.tune_specs()[:1], [0], TrainSettings(epochs=1), exp_dir=tmp_path / "exp",
                 ckpt_dir=tmp_path / "ck", evaluate_test=False, log=lambda s: None)
    tuned = tmp_path / "exp" / "tune" / "datf_lh0.1_lm0.0" / "seed_0" / "run.json"
    reuse = {("datf_full", 0): {"ckpt": tmp_path / "ck" / "tune" / "datf_lh0.1_lm0.0_seed0.pt",
                                "history": json.loads(tuned.read_text())["history"]}}
    spec = [s for s in ex.final_specs(0.1, 0.0) if s.name == "datf_full"]
    ex.run_stage("final", spec, [0], TrainSettings(epochs=1), exp_dir=tmp_path / "exp", ckpt_dir=tmp_path / "ck",
                 evaluate_test=True, reuse=reuse, log=lambda s: None)
    d = tmp_path / "exp" / "final" / "datf_full" / "seed_0"
    rec = json.loads((d / "run.json").read_text())
    assert (d / "test_predictions.csv").exists() and rec["test"]["n_engines"] == 100 and "reused" in rec["note"]
