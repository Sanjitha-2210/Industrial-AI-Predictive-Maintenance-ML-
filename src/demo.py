"""SIMULATED engine trajectories for demo mode. Never real data; never used for metrics.

Method (data-driven, stage-aware):
  1. From the real FD001 *training* file, compute for every sensor channel the mean reading as a function
     of true remaining useful life r (0 = failure), lightly smoothed. Engines older than the oldest
     well-populated r use the healthy plateau value.
  2. Estimate two noise levels per channel from the same data: the engine-to-engine offset spread and the
     cycle-to-cycle spread around each engine's own profile.
  3. A simulated engine is defined by its scenario RUL at the last observed cycle (drawn from the chosen
     stage's band, reproducibly from the seed). Cycle c of n has scenario RUL = rul_end + (n - c); its
     readings are profile(RUL) + one engine offset + per-cycle noise.
So the sensors and the scenario RUL are consistent by construction. The scenario RUL is a property of the
simulation, not a measured ground truth, and must never be mixed with real-data metrics.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import RAW_DIR, RUL_CLIP
from src.data.load_data import COLUMNS, OP_COLS, SENSOR_COLS, dataset_available, load_train
from src.data.table import Table, groups

SIMULATED_LABEL = "SIMULATED DATA - not from the NASA C-MAPSS dataset"

# Scenario RUL at the last observed cycle, per simulation scenario (cycles). These fixed bands define the demo
# scenarios only (values unchanged: 80-110 / 33-42 / 5-16); they are NOT health-state thresholds. Health states
# for simulated engines come from src/health_state.py like for any other engine.
STAGE_RUL_BANDS = {
    "normal": (80, 110),     # early-degradation scenario
    "warning": (33, 42),     # late-degradation scenario
    "critical": (5, 16),     # end-of-life scenario
}
STAGES = tuple(STAGE_RUL_BANDS)
MIN_ENGINES_PER_RUL = 10   # r values seen in fewer training engines are treated as the healthy plateau
SMOOTH_WINDOW = 9          # centred moving average over r (cycles)


class DegradationProfile:
    """Per-channel mean reading vs true RUL, plus noise levels, estimated from real training data."""

    def __init__(self, train: Table):
        self.varying = [c for c in OP_COLS + SENSOR_COLS if np.std(train[c]) > 1e-9]
        self.constants = {c: float(train[c][0]) for c in OP_COLS + SENSOR_COLS if c not in self.varying}
        last = np.zeros(len(train["unit"]), dtype=np.int64)
        for _, idx in groups(train):
            last[idx] = train["cycle"][idx].max()
        rul = last - train["cycle"]
        X = np.column_stack([train[c] for c in self.varying]).astype(float)

        counts = np.bincount(rul)
        sums = np.zeros((len(counts), X.shape[1]))
        np.add.at(sums, rul, X)
        self.r_top = int(np.flatnonzero(counts >= MIN_ENGINES_PER_RUL).max())
        mean = sums[: self.r_top + 1] / counts[: self.r_top + 1, None]
        k = SMOOTH_WINDOW // 2
        padded = np.vstack([np.repeat(mean[:1], k, 0), mean, np.repeat(mean[-1:], k, 0)])
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        self.mean = np.column_stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(X.shape[1])])

        res = X - self.mean[np.minimum(rul, self.r_top)]
        offsets = np.array([res[idx].mean(axis=0) for _, idx in groups(train)])
        engine_of_row = np.concatenate([np.full(len(idx), i) for i, (_, idx) in enumerate(groups(train))])
        order = np.concatenate([idx for _, idx in groups(train)])
        within = res[order] - offsets[engine_of_row]
        self.between_std = offsets.std(axis=0, ddof=1)
        self.within_std = within.std(axis=0, ddof=1)

    def at(self, rul: np.ndarray) -> np.ndarray:
        return self.mean[np.minimum(np.asarray(rul), self.r_top)]


@lru_cache(maxsize=4)
def _profile_cached(raw_dir: str) -> DegradationProfile:
    return DegradationProfile(load_train(Path(raw_dir)))


def get_profile(raw_dir=RAW_DIR) -> DegradationProfile | None:
    """Profile from train_FD001.txt, or None if the dataset is not available."""
    return _profile_cached(str(raw_dir)) if dataset_available(raw_dir) else None


def scenario_rul_end(stage: str, seed: int) -> int:
    lo, hi = STAGE_RUL_BANDS[stage]
    return int(np.random.default_rng([seed, STAGES.index(stage)]).integers(lo, hi + 1))


def simulate_engine(n_cycles: int = 180, seed: int = 0, stage: str = "normal",
                    profile: DegradationProfile | None = None, unit: int = 999) -> tuple[Table, dict]:
    """Returns (readings table, info). info has the stage, scenario RUL per cycle and the method used."""
    stage = stage.lower()
    if stage not in STAGE_RUL_BANDS:
        raise ValueError(f"stage must be one of {STAGES}")
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    rul_end = scenario_rul_end(stage, seed)
    cycles = np.arange(1, n_cycles + 1, dtype=np.int64)
    scenario_rul = rul_end + (n_cycles - cycles)
    rng = np.random.default_rng(seed)
    data = {"unit": np.full(n_cycles, unit, dtype=np.int64), "cycle": cycles}
    if profile is not None:
        offset = rng.normal(0.0, profile.between_std)
        values = profile.at(scenario_rul) + offset + rng.normal(0.0, profile.within_std, (n_cycles, len(profile.varying)))
        for j, c in enumerate(profile.varying):
            data[c] = values[:, j]
        for c, v in profile.constants.items():
            data[c] = np.full(n_cycles, v)
        method = ("FD001 training-set mean degradation profile vs RUL + engine offset + cycle noise "
                  "(estimated from the real training data)")
    else:
        # dataset unavailable: generic synthetic curves; not calibrated to the model
        wear = np.clip(1 - scenario_rul / RUL_CLIP, 0, 1) ** 2
        for c in OP_COLS + SENSOR_COLS:
            data[c] = 100 + 10 * (rng.normal(0, 1) * wear + rng.normal(0, 0.4, n_cycles))
        method = "generic synthetic curves (training data unavailable; not calibrated to the model)"
    info = {"stage": stage, "seed": seed, "n_cycles": n_cycles, "scenario_rul_at_last_cycle": rul_end,
            "scenario_rul": scenario_rul, "stage_band": STAGE_RUL_BANDS[stage], "method": method}
    return {c: data[c] for c in COLUMNS}, info
