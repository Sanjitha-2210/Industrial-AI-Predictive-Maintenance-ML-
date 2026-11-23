"""Information-matched LSTM control: same architecture as the baseline LSTM, input [x, x - b]."""
import numpy as np
import torch

from src.models.datf_net import count_parameters
from src.models.lstm import LSTMRegressor
from src.models.lstm_control import BaselineInfoLSTM


def test_architecture_equals_baseline_lstm_with_doubled_input():
    ctrl, base = BaselineInfoLSTM(17), LSTMRegressor(17)
    ref = LSTMRegressor(34)
    assert [p.shape for p in ctrl.parameters()] == [p.shape for p in ref.parameters()]
    # only the first LSTM layer's input weights grow (17 -> 34 inputs): 4 gates x 64 units x 17 extra inputs
    assert count_parameters(ctrl) == count_parameters(base) + 4 * 64 * 17 == 65_153
    assert ctrl.lstm_model.lstm.num_layers == base.lstm.num_layers and ctrl.lstm_model.lstm.hidden_size == 64


def test_input_is_window_plus_baseline_deviation():
    x, b = torch.randn(3, 30, 17), torch.randn(3, 17)
    z = BaselineInfoLSTM.build_input(x, b)
    assert z.shape == (3, 30, 34)
    assert torch.equal(z[..., :17], x) and torch.allclose(z[..., 17:], x - b[:, None, :])


def test_output_depends_on_baseline_and_requires_it():
    m = BaselineInfoLSTM(17).eval()
    x = torch.randn(2, 30, 17)
    assert m(x, torch.zeros(2, 17)).shape == (2,)
    assert not torch.allclose(m(x, torch.zeros(2, 17)), m(x, torch.ones(2, 17)))
    try:
        m(x, None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_training_deterministic_and_checkpoint_roundtrip(tmp_path):
    from src.train_datf import RunSpec, TrainSettings, load_checkpoint, predict, train_run
    rng = np.random.default_rng(0)
    n = 64
    y = rng.uniform(0, 125, n).astype(np.float32)
    d = {"X": rng.normal(size=(n, 30, 17)).astype(np.float32), "y": y, "b": rng.normal(size=(n, 17)).astype(np.float32),
         "hi_target": np.ones((n, 30), np.float32), "unit": np.repeat(np.arange(8), 8)}
    spec, st = RunSpec("lstm_info_matched", "lstm_info"), TrainSettings(epochs=2)
    h1 = train_run(spec, d, d, 3, st, 125.0, tmp_path / "a.pt", torch.device("cpu"), log=lambda s: None)
    h2 = train_run(spec, d, d, 3, st, 125.0, tmp_path / "b.pt", torch.device("cpu"), log=lambda s: None)
    assert h1["val_loss"] == h2["val_loss"] and h1["train_total_loss"] == h1["train_loss"]  # no auxiliary loss
    m, ck = load_checkpoint(tmp_path / "a.pt", 17, 30, torch.device("cpu"))
    assert isinstance(m, BaselineInfoLSTM) and ck["model_type"] == "lstm_info"
    p1, hi = predict(m, d["X"], d["b"], 125.0, torch.device("cpu"))
    p2, _ = predict(m, d["X"], d["b"] + 1.0, 125.0, torch.device("cpu"))
    assert hi is None and p1.shape == (n,) and not np.allclose(p1, p2)


def test_control_uses_baseline_hyperparameters_without_tuning():
    from src.experiments import control_specs
    (spec,) = control_specs()
    assert spec.model_type == "lstm_info" and not spec.lambda_hi and not spec.lambda_mono and spec.datf == {}
