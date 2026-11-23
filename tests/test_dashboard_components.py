"""Pure HTML builders of the dashboard's UI components (no Streamlit session needed)."""
import re
import sys

from src import config as C

sys.path.insert(0, str(C.ROOT / "dashboard"))
from components import (metric_card_html, prediction_card_html, sensor_list_html, state_strip_html,  # noqa: E402
                        status_badge_html,
                        trend_arrow, uncertainty_range_html)


def test_status_badge_has_text_label_not_colour_only():
    for s in ("Insufficient History", "Healthy / Baseline-consistent", "Degrading", "Boundary / Uncertain",
              "Near Failure"):
        h = status_badge_html(s)
        assert s in h and "color:" in h
    assert "N/A" in status_badge_html(None)


def test_metric_card_escapes_and_exposes_value():
    h = metric_card_html("Engine", "<b>34</b>", unit="cycles", sub="a & b")
    assert "&lt;b&gt;34&lt;/b&gt;" in h and "a &amp; b" in h and "<b>34</b>" not in h
    assert re.search(r"data-kpi='Engine' data-value='&lt;b&gt;34&lt;/b&gt;'", h)


def test_uncertainty_range_bounds_and_clipping():
    h = uncertainty_range_html(10.0, 15.0, 125)
    assert "Lower <b>0.0</b>" in h and "Upper <b>25.0</b>" in h and "Prediction <b>10.0</b>" in h


def test_state_strip_shows_runs_in_order():
    states = ["Insufficient History"] * 29 + ["Healthy / Baseline-consistent"] * 50 + ["Degrading"] * 20 + ["Near Failure"] * 10
    h = state_strip_html(states, list(range(1, len(states) + 1)))
    assert h.count("class='strip-seg'") == 4 and "cycles 1–29" in h and "cycles 100–109" in h and "(now)" in h


def test_prediction_card_keeps_current_prediction_when_latched():
    h = prediction_card_html(51.0, "Near Failure", 3.0, "LSTM baseline", latched=True, latch_cycle=150,
                             above_boundary=True, boundary=45)
    assert "data-value='51'" in h and "Near Failure — latched" in h and "Latched at cycle <b>150</b>" in h
    assert "current prediction is above boundary (45)" in h


def test_sensor_trend_arrows():
    assert trend_arrow(1.2, 0.5) == ("↑", "Increasing")
    assert trend_arrow(-0.9, 0.5) == ("↓", "Decreasing")
    assert trend_arrow(0.2, 0.5) == ("→", "Stable")
    h = sensor_list_html([{"sensor": "sensor_11", "drift_std_units": 2.34, "direction": "rising"}])
    assert "sensor_11" in h and "Increasing" in h and "+2.3σ" in h
    assert "No sensor trend data" in sensor_list_html([])
