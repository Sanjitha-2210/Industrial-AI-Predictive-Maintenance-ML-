"""Cross-validation of the data-driven health-state methodology (VALIDATION ONLY - writes only new cv_* files here).

Run from the project root:  .\\.venv\\Scripts\\python.exe outputs\\health_threshold_analysis\\cv_validation.py

* 5-fold ENGINE-level CV on the 80 training engines: every sensor parameter (PCA weights, noise level, baseline
  limit L1, persistence m) and the acceleration boundary R_acc are re-derived on the 64 fold-training engines and
  evaluated on the 16 held-out engines. No model is involved in these parameters.
* The model-based Near-Failure rule needs predicted RUL. Predictions on the 80 training engines are in-sample for
  the existing models, so it is evaluated on the 20 VALIDATION engines (held out when the models were trained),
  with the deployed LSTM baseline (primary) and DATF-Net (secondary). Inference only; nothing is retrained.
* The official TEST set is never loaded.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.load_data import SENSOR_COLS, load_train  # noqa: E402
from src.data.preprocess import Preprocessor  # noqa: E402
from src.data.table import groups, matrix  # noqa: E402

OUT = Path(__file__).resolve().parent
BASE, W, K_SPC = 10, 10, 3.0
M_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40]   # persistence candidates (cycles)
PREMATURE_TOL = 0.05   # policy tolerance to pick m: <= 5 % of fold-training engines declared before their change point
KNEE_TOL = 0.01        # fallback if no m meets the tolerance: smallest m within 1 point of the lowest premature rate
STRONG = 20            # "strongly premature": declared more than 20 cycles before the sensor change point
N_FOLDS, CV_SEED, N_BOOT = 5, 2024, 2000
C_BLUE, C_ORANGE, C_AQUA, C_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8980"
S_GOOD, S_WARN, S_CRIT, S_UNC = "#0ca30c", "#fab219", "#d03b3b", "#7f7f7f"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8980",
                     "axes.titlesize": 10, "legend.frameon": False})
# This script writes only cv_* files and plots 10-14; the earlier analysis files are never touched.


def save_fig(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig)


def causal_mean(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0))
    t = np.arange(len(x))
    lo = np.maximum(0, t - w + 1)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


def first_sustained(mask, persist):
    n = len(mask)
    run = 0
    for i in range(n):
        run = run + 1 if mask[i] else 0
        if run >= persist:
            return i - persist + 1
    # a run still in progress when the record ends (engine failed) counts, as in the exploratory analysis
    return n - run if run > 0 else None


def fit_hinge_quad(y):
    t = np.arange(len(y), dtype=float)
    best = (np.inf, None)
    for tau in range(BASE, len(y) - 10):
        h = np.maximum(0.0, t - tau) ** 2
        A = np.column_stack([np.ones_like(t), h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((y - A @ coef) ** 2).sum())
        if sse < best[0]:
            best = (sse, tau)
    return best


def fit_two_break(y, step=3):
    t = np.arange(len(y), dtype=float); n = len(y)
    best = (np.inf, None, None)
    grid = np.arange(BASE, n - 10, step)
    for t1 in grid:
        h1 = np.maximum(0.0, t - t1)
        t2s = grid[grid > t1 + 5]
        if len(t2s) == 0:
            continue
        H2 = np.maximum(0.0, t[:, None] - t2s[None, :])
        G = np.empty((len(t2s), 3, 3)); r = np.empty((len(t2s), 3))
        G[:, 0, 0] = n; G[:, 0, 1] = G[:, 1, 0] = h1.sum(); G[:, 1, 1] = (h1 * h1).sum()
        G[:, 0, 2] = G[:, 2, 0] = H2.sum(0); G[:, 1, 2] = G[:, 2, 1] = (h1[:, None] * H2).sum(0); G[:, 2, 2] = (H2 * H2).sum(0)
        r[:, 0] = y.sum(); r[:, 1] = (h1 * y).sum(); r[:, 2] = (H2 * y[:, None]).sum(0)
        coef = np.linalg.solve(G, r[..., None])[..., 0]
        pred = coef[:, 0][None, :] + coef[:, 1][None, :] * h1[:, None] + coef[:, 2][None, :] * H2
        sse = ((y[:, None] - pred) ** 2).sum(0)
        k = int(np.argmin(sse))
        if sse[k] < best[0]:
            best = (float(sse[k]), int(t1), int(t2s[k]))
    return best


def bic(sse, n, k):
    return n * np.log(sse / n) + k * np.log(n)


def dist(v):
    v = np.array([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "min": float(v.min()), "q10": float(np.percentile(v, 10)), "q25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q75": float(np.percentile(v, 75)), "q90": float(np.percentile(v, 90)),
            "max": float(v.max()), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


# ================= data =================
prep = Preprocessor.load()                          # read-only
raw = load_train()
val_units = sorted(int(u) for u in prep.val_units)
all_units = sorted(int(u) for u in np.unique(raw["unit"]))
train_units = [u for u in all_units if u not in set(val_units)]
sens = [c for c in prep.feature_cols if c in SENSOR_COLS]
X = matrix(prep.transform(raw), sens)
eng = {u: idx[np.argsort(raw["cycle"][idx])] for u, idx in groups(raw)}
dev = {u: X[eng[u]] - X[eng[u]][:BASE].mean(0) for u in all_units}
life = {u: len(eng[u]) for u in all_units}


# ================= parameter discovery on a set of engines =================
def derive(units: list[int]) -> dict:
    P = np.concatenate([dev[u] for u in units])
    ev, V = np.linalg.eigh(np.cov(P, rowvar=False))
    w = V[:, np.argmax(ev)]
    if np.corrcoef(P @ w, np.concatenate([np.arange(life[u]) for u in units]))[0, 1] < 0:
        w = -w
    DI = {u: dev[u] @ w for u in all_units}
    DIs = {u: causal_mean(DI[u], W) for u in all_units}
    sigma_w = float(np.concatenate([DIs[u][W - 1:BASE + 20] - DI[u][:BASE].mean() for u in units]).std(ddof=1))
    L1 = K_SPC * sigma_w
    # per-engine retrospective references (sensor-only): onset change point and acceleration point
    ref = {}
    for u in all_units:
        y = DI[u]; n = len(y)
        s_q, tau = fit_hinge_quad(y)
        s_2, t1, t2 = fit_two_break(y)
        ref[u] = {"tau_onset": tau, "tau_acc": t2, "acc_supported": bool(bic(s_2, n, 6) - bic(s_q, n, 4) < -10)}
    # persistence: smallest m with <= PREMATURE_TOL of fold-training engines flagged before their change point
    m_table = {}
    for m in M_GRID:
        flags = [declare(first_sustained(DIs[u][BASE:] > L1, m), m, life[u]) for u in units]
        prem = np.mean([f is not None and f < ref[u]["tau_onset"] for f, u in zip(flags, units)])
        delay = [f - ref[u]["tau_onset"] for f, u in zip(flags, units) if f is not None]
        m_table[m] = {"premature_rate": float(prem), "median_delay": float(np.median(delay)), "detected": int(sum(f is not None for f in flags))}
    meets = [m for m in M_GRID if m_table[m]["premature_rate"] <= PREMATURE_TOL]
    if meets:
        m_sel, m_rule = meets[0], f"premature <= {PREMATURE_TOL}"
    else:
        lowest = min(v["premature_rate"] for v in m_table.values())
        m_sel = next(m for m in M_GRID if m_table[m]["premature_rate"] <= lowest + KNEE_TOL)
        m_rule = f"fallback: no m met {PREMATURE_TOL}; smallest m within {KNEE_TOL} of lowest rate {lowest:.3f}"
    sup = [life[u] - 1 - ref[u]["tau_acc"] for u in units if ref[u]["acc_supported"]]
    return {"w": w, "DI": DI, "DIs": DIs, "sigma_w": sigma_w, "L1": L1, "ref": ref, "m_table": m_table, "m": m_sel, "m_rule": m_rule,
            "R_acc": float(np.median(sup)), "n_acc_supported": len(sup), "acc_ruls": sup}


def declare(start, m, n):
    """Cycle at which Degrading can be DECLARED online: end of the m-cycle run (or the last cycle if the record ends)."""
    return None if start is None else min(start + BASE + m - 1, n - 1)


def onset_flag(p, u, m):
    return declare(first_sustained(p["DIs"][u][BASE:] > p["L1"], m), m, life[u])


# ================= full-data (80 engines) reference =================
full = derive(train_units)
print(f"full: L1 {full['L1']:.3f}  m {full['m']}  R_acc {full['R_acc']:.1f} ({full['n_acc_supported']} supported)", flush=True)

# ================= 5-fold engine-level CV =================
rng = np.random.default_rng(CV_SEED)
perm = rng.permutation(train_units)
folds = [sorted(int(u) for u in perm[i::N_FOLDS]) for i in range(N_FOLDS)]
fold_rows, heldout_rows, m_curves = [], [], []
for k, test_f in enumerate(folds):
    tr_f = [u for u in train_units if u not in set(test_f)]
    p = derive(tr_f)
    m_curves.append({m: p["m_table"][m] for m in M_GRID})
    cos = float(abs(np.dot(p["w"], full["w"])) / (np.linalg.norm(p["w"]) * np.linalg.norm(full["w"])))
    onset_rul, delays, prem, first30, dips, acc_err = [], [], 0, 0, [], []
    for u in test_f:
        f = onset_flag(p, u, p["m"])
        ref = p["ref"][u]
        rec = {"fold": k + 1, "engine": u, "life": life[u], "L1_fold": p["L1"], "m_fold": p["m"], "R_acc_fold": p["R_acc"],
               "onset_flag_cycle": f, "rul_at_onset_flag": None if f is None else life[u] - 1 - f,
               "sensor_change_point_cycle": ref["tau_onset"], "delay_vs_change_point": None if f is None else f - ref["tau_onset"],
               "premature": bool(f is not None and f < ref["tau_onset"]), "flag_in_first_30_cycles": bool(f is not None and f < 30),
               "frac_cycles_below_L1_after_flag": None if f is None else float(np.mean(p["DIs"][u][f:] <= p["L1"])),
               "acc_supported": ref["acc_supported"], "rul_at_own_acceleration": life[u] - 1 - ref["tau_acc"],
               "R_acc_error": (p["R_acc"] - (life[u] - 1 - ref["tau_acc"])) if ref["acc_supported"] else None}
        f_full = onset_flag(full, u, full["m"])
        rec["rul_at_onset_flag_full_params"] = None if f_full is None else life[u] - 1 - f_full
        rec["strongly_premature"] = bool(f is not None and f < ref["tau_onset"] - STRONG)
        for mm in (5, 10, 20, 30):
            g = onset_flag(p, u, mm)
            rec[f"m{mm}_delay"] = None if g is None else g - ref["tau_onset"]
            rec[f"m{mm}_rul_at_flag"] = None if g is None else life[u] - 1 - g
        heldout_rows.append(rec)
    H = [r for r in heldout_rows if r["fold"] == k + 1]
    fold_rows.append({"fold": k + 1, "n_train": len(tr_f), "n_heldout": len(test_f), "pca_cosine_vs_full": cos,
                      "sigma_w": p["sigma_w"], "L1": p["L1"], "m_selected": p["m"], "m_rule": p["m_rule"], "R_acc": p["R_acc"],
                      "n_acc_supported_train": p["n_acc_supported"],
                      "heldout_detected": sum(r["onset_flag_cycle"] is not None for r in H),
                      "heldout_premature": sum(r["premature"] for r in H),
                      "heldout_strongly_premature": sum(r["strongly_premature"] for r in H),
                      "heldout_flag_first_30": sum(r["flag_in_first_30_cycles"] for r in H),
                      "heldout_median_rul_at_onset": float(np.median([r["rul_at_onset_flag"] for r in H if r["rul_at_onset_flag"] is not None])),
                      "heldout_median_delay": float(np.median([r["delay_vs_change_point"] for r in H if r["delay_vs_change_point"] is not None])),
                      "heldout_acc_supported": sum(r["acc_supported"] for r in H),
                      "heldout_median_abs_R_acc_error": float(np.median([abs(r["R_acc_error"]) for r in H if r["R_acc_error"] is not None]))})
    print(f"fold {k + 1}: L1 {p['L1']:.3f} m {p['m']} R_acc {p['R_acc']:.1f} cos {cos:.4f}", flush=True)

# ================= engine bootstrap CIs (80 engines, full-data weights) =================
brng = np.random.default_rng(7)
early = {u: full["DIs"][u][W - 1:BASE + 20] - full["DI"][u][:BASE].mean() for u in train_units}
onset_full = {u: onset_flag(full, u, full["m"]) for u in train_units}
acc_full = {u: (life[u] - 1 - full["ref"][u]["tau_acc"]) for u in train_units if full["ref"][u]["acc_supported"]}
bL1, bR, bOn = [], [], []
for _ in range(N_BOOT):
    s = brng.choice(train_units, len(train_units), replace=True)
    bL1.append(K_SPC * np.concatenate([early[u] for u in s]).std(ddof=1))
    accs = [acc_full[u] for u in s if u in acc_full]
    bR.append(np.median(accs))
    bOn.append(np.median([life[u] - 1 - onset_full[u] for u in s if onset_full[u] is not None]))
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731

# ================= Near-Failure / combined states on the 20 validation engines (inference only) =================
import torch  # noqa: E402

from src.data.degradation import load_train_val  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.train_datf import load_checkpoint, predict as datf_predict  # noqa: E402

pred = Predictor()                                     # deployed LSTM baseline (read-only)
acc_q10, acc_q90 = float(np.percentile(full["acc_ruls"], 10)), float(np.percentile(full["acc_ruls"], 90))
val_tables = {}
for u in val_units:
    idx = eng[u]
    t = {c: raw[c][idx] for c in raw}
    Wn, _ = pred._windows(t, pad_all=True)
    mean, std = pred._predict_windows(Wn, mc_samples=30)
    val_tables[u] = {"pred": mean, "unc": std}
_, vaw = load_train_val(prep)
dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datf_preds = []
for ck in sorted((ROOT / "models" / "checkpoints" / "datf" / "final").glob("datf_full_seed*.pt")):
    mdl, _ = load_checkpoint(ck, len(prep.feature_cols), prep.seq_len, dev_t)
    r_, _ = datf_predict(mdl, vaw["X"], vaw["b"], float(prep.rul_clip), dev_t)
    datf_preds.append(r_)
datf_mean = np.mean(datf_preds, axis=0)


def near_failure_eval(pred_by_engine: dict, R: float, label: str) -> dict:
    """pred_by_engine[u] = (cycle_index_array, predicted_rul_array)."""
    lead, missed, early_n, late_n, flips = [], 0, 0, 0, []
    tp = fp = fn = tn = 0
    for u, (cyc, pr) in pred_by_engine.items():
        true = life[u] - 1 - cyc
        flag = pr <= R
        first = np.flatnonzero(flag)
        if len(first) == 0:
            missed += 1
        else:
            lr = int(true[first[0]]); lead.append(lr)
            early_n += lr > acc_q90; late_n += lr < acc_q10
        flips.append(int(np.sum(np.diff(flag.astype(int)) == -1)))   # Near Failure -> back to non-near
        tt = true <= R
        tp += int((flag & tt).sum()); fp += int((flag & ~tt).sum()); fn += int((~flag & tt).sum()); tn += int((~flag & ~tt).sum())
    return {"model": label, "R_acc": R, "engines": len(pred_by_engine), "missed_engines": missed,
            "true_rul_at_first_flag": dist(lead), "early_flags_true_rul_gt_train_q90": int(early_n),
            "late_flags_true_rul_lt_train_q10": int(late_n), "train_acc_q10_q90": [acc_q10, acc_q90],
            "back_transitions_per_engine_unlatched": dist(flips),
            "cycle_level": {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}}


lstm_by = {u: (np.arange(life[u]), val_tables[u]["pred"]) for u in val_units}
datf_by = {}
for u in val_units:
    m = vaw["unit"] == u
    datf_by[u] = (vaw["end_cycle"][m] - 1, datf_mean[m])       # end_cycle is 1-based cycle number
nf = {"lstm_full_R": near_failure_eval(lstm_by, full["R_acc"], "LSTM baseline (deployed)"),
      "datf_full_R": near_failure_eval(datf_by, full["R_acc"], "DATF-Net (mean of 5 seeds)"),
      "lstm_by_fold_R": [near_failure_eval(lstm_by, fr["R_acc"], f"LSTM, fold {fr['fold']} R_acc") for fr in fold_rows]}

# combined state timeline (final definition) on validation engines, full-data frozen parameters
STATE_NAMES = ["Insufficient history", "Healthy", "Degrading", "Near Failure", "Boundary / Uncertain"]
state_rows, share = [], {s: [] for s in STATE_NAMES}
R_int = int(np.floor(full["R_acc"]))
bins = [(125, 1e9), (R_int + 1, 124), (0, R_int)]
share_by_bin = {f"{a}-{b if b < 1e8 else 'inf'}": {s: 0 for s in STATE_NAMES} for a, b in bins}
disagree_cycles, boundary_cycles, total_cycles = 0, 0, 0
for u in val_units:
    n = life[u]
    f = onset_flag(full, u, full["m"])
    pr, un = val_tables[u]["pred"], val_tables[u]["unc"]
    states = []
    for c in range(n):
        if c < BASE:
            s = "Insufficient history"
        else:
            degr = f is not None and c >= f
            near = pr[c] <= full["R_acc"]
            close = abs(pr[c] - full["R_acc"]) <= un[c]
            disagree = near and not degr
            if close or disagree:
                s = "Boundary / Uncertain"; boundary_cycles += 1; disagree_cycles += int(disagree)
            elif near:
                s = "Near Failure"
            elif degr:
                s = "Degrading"
            else:
                s = "Healthy"
            total_cycles += 1
        states.append(s)
        true = n - 1 - c
        for (a, b) in bins:
            if a <= true <= b:
                share_by_bin[f"{a}-{b if b < 1e8 else 'inf'}"][s] += 1
    for s in STATE_NAMES:
        share[s].append(states.count(s) / n)
    back = sum(1 for i in range(1, n) if STATE_NAMES.index(states[i]) < STATE_NAMES.index(states[i - 1])
               and states[i - 1] != "Boundary / Uncertain" and states[i] != "Boundary / Uncertain")
    state_rows.append({"engine": u, "life": n, "onset_flag_cycle": f, "rul_at_onset_flag": None if f is None else n - 1 - f,
                       **{f"share_{s.split(' ')[0].lower()}": states.count(s) / n for s in STATE_NAMES},
                       "backward_state_changes": back, "states": states})

results = {
    "protocol": {"folds": N_FOLDS, "cv_seed": CV_SEED, "fold_engines": folds, "k_spc": K_SPC, "smoothing_window": W,
                 "persistence_grid": M_GRID, "persistence_rule": f"smallest m with premature rate <= {PREMATURE_TOL} on fold-training engines (policy tolerance)",
                 "near_failure_eval_engines": val_units, "test_set": "not loaded"},
    "full_80": {"sigma_w": full["sigma_w"], "L1": full["L1"], "m": full["m"], "m_rule": full["m_rule"], "R_acc": full["R_acc"],
                "n_acc_supported": full["n_acc_supported"], "persistence_table": full["m_table"]},
    "folds": fold_rows,
    "bootstrap_ci95": {"L1": ci(bL1), "R_acc": ci(bR), "median_rul_at_onset": ci(bOn), "n_boot": N_BOOT},
    "heldout_summary": {
        "onset_rul": dist([r["rul_at_onset_flag"] for r in heldout_rows]),
        "delay_vs_change_point": dist([r["delay_vs_change_point"] for r in heldout_rows]),
        "detected": sum(r["onset_flag_cycle"] is not None for r in heldout_rows), "n": len(heldout_rows),
        "premature": sum(r["premature"] for r in heldout_rows),
        "strongly_premature_gt20_cycles": sum(r["strongly_premature"] for r in heldout_rows),
        "persistence_comparison_heldout": {f"m{mm}": {
            "premature": sum(1 for r in heldout_rows if r[f"m{mm}_delay"] is not None and r[f"m{mm}_delay"] < 0),
            "strongly_premature": sum(1 for r in heldout_rows if r[f"m{mm}_delay"] is not None and r[f"m{mm}_delay"] < -STRONG),
            "delay": dist([r[f"m{mm}_delay"] for r in heldout_rows]), "rul_at_flag": dist([r[f"m{mm}_rul_at_flag"] for r in heldout_rows]),
            "detected": sum(1 for r in heldout_rows if r[f"m{mm}_delay"] is not None)} for mm in (5, 10, 20, 30)},
        "flag_in_first_30_cycles": sum(r["flag_in_first_30_cycles"] for r in heldout_rows),
        "abs_onset_shift_fold_vs_full_params": dist([abs(r["rul_at_onset_flag"] - r["rul_at_onset_flag_full_params"])
                                                     for r in heldout_rows if r["rul_at_onset_flag"] is not None and r["rul_at_onset_flag_full_params"] is not None]),
        "frac_cycles_below_L1_after_flag": dist([r["frac_cycles_below_L1_after_flag"] for r in heldout_rows]),
        "R_acc_error_supported_engines": dist([r["R_acc_error"] for r in heldout_rows]),
        "abs_R_acc_error": dist([abs(r["R_acc_error"]) for r in heldout_rows if r["R_acc_error"] is not None]),
        "acc_supported": sum(r["acc_supported"] for r in heldout_rows)},
    "persistence_by_fold": [{str(m): v for m, v in c.items()} for c in m_curves],
    "near_failure_validation_engines": nf,
    "combined_states_validation": {"state_share_per_engine": {s: dist(v) for s, v in share.items()},
                                   "state_counts_by_true_rul_bin": share_by_bin,
                                   "boundary_cycles": boundary_cycles, "of_which_disagreement": disagree_cycles,
                                   "classified_cycles": total_cycles,
                                   "backward_state_changes_per_engine": dist([r["backward_state_changes"] for r in state_rows])},
}
(OUT / "cv_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
with open(OUT / "cv_fold_summary.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(fold_rows[0])); wtr.writeheader(); wtr.writerows(fold_rows)
with open(OUT / "cv_heldout_engines.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(heldout_rows[0])); wtr.writeheader(); wtr.writerows(heldout_rows)
with open(OUT / "cv_validation_engine_states.csv", "w", newline="") as f:
    keys = [k for k in state_rows[0] if k != "states"]
    wtr = csv.DictWriter(f, fieldnames=keys); wtr.writeheader(); wtr.writerows([{k: r[k] for k in keys} for r in state_rows])

# ================= plots =================
fr = fold_rows
fig, axs = plt.subplots(1, 4, figsize=(13, 3))
for ax, key, title, b in ((axs[0], "L1", "Baseline limit L1 (σ)", bL1), (axs[1], "m_selected", "Persistence m (cycles)", None),
                          (axs[2], "R_acc", "Acceleration boundary R_acc (RUL)", bR),
                          (axs[3], "heldout_median_rul_at_onset", "Held-out median RUL at onset", bOn)):
    ax.plot([r["fold"] for r in fr], [r[key] for r in fr], "o", color=C_BLUE, ms=7, label="Fold estimate")
    full_v = {"L1": full["L1"], "m_selected": full["m"], "R_acc": full["R_acc"],
              "heldout_median_rul_at_onset": float(np.median([life[u] - 1 - onset_full[u] for u in train_units if onset_full[u] is not None]))}[key]
    ax.axhline(full_v, color=C_ORANGE, lw=1.2, label="All 80 engines")
    if b is not None:
        ax.axhspan(*ci(b), color=C_ORANGE, alpha=0.15, label="Bootstrap 95% CI")
    ax.set(title=title, xlabel="Fold", xticks=range(1, N_FOLDS + 1))
axs[0].legend(fontsize=7)
save_fig(fig, "10_cv_parameter_stability.png")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
mc = np.array([[c[m]["premature_rate"] for m in M_GRID] for c in m_curves])
md = np.array([[c[m]["median_delay"] for m in M_GRID] for c in m_curves])
ax.plot(M_GRID, mc.mean(0) * 100, marker="o", color=S_CRIT, label="Premature flags (% of engines)")
ax.fill_between(M_GRID, mc.min(0) * 100, mc.max(0) * 100, color=S_CRIT, alpha=0.15)
ax.axhline(PREMATURE_TOL * 100, color=S_CRIT, ls=":", lw=1)
ax2 = ax.twinx()
ax2.plot(M_GRID, md.mean(0), marker="s", color=C_BLUE, label="Median detection delay (cycles)")
ax2.fill_between(M_GRID, md.min(0), md.max(0), color=C_BLUE, alpha=0.12)
ax.set(title="Persistence trade-off (fold-training engines; band = range over folds)", xlabel="Persistence m (consecutive cycles)",
       ylabel="Premature flags (%)"); ax2.set_ylabel("Delay after sensor change point (cycles)")
ax2.spines["right"].set_visible(True); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center")
save_fig(fig, "11_cv_persistence_tradeoff.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
data = [[r["rul_at_onset_flag"] for r in heldout_rows if r["fold"] == k + 1 and r["rul_at_onset_flag"] is not None] for k in range(N_FOLDS)]
axs[0].boxplot(data, tick_labels=[str(k + 1) for k in range(N_FOLDS)], patch_artist=True,
               boxprops=dict(facecolor="#cfe0f6", color=C_BLUE), medianprops=dict(color=C_BLUE))
axs[0].set(title="Held-out engines: RUL when Degrading is flagged", xlabel="Fold", ylabel="True RUL at flag (cycles)")
errs = [[r["R_acc_error"] for r in heldout_rows if r["fold"] == k + 1 and r["R_acc_error"] is not None] for k in range(N_FOLDS)]
axs[1].boxplot(errs, tick_labels=[str(k + 1) for k in range(N_FOLDS)], patch_artist=True,
               boxprops=dict(facecolor="#fbd9c9", color=C_ORANGE), medianprops=dict(color=C_ORANGE))
axs[1].axhline(0, color=C_GREY, lw=1)
axs[1].set(title="Fold R_acc minus held-out engine's own acceleration RUL", xlabel="Fold", ylabel="Error (cycles)")
save_fig(fig, "12_cv_heldout_onset_and_acceleration.png")

cols = {"Insufficient history": "#d9d8d2", "Healthy": S_GOOD, "Degrading": S_WARN, "Near Failure": S_CRIT, "Boundary / Uncertain": S_UNC}
fig, ax = plt.subplots(figsize=(9, 5.2))
for i, r in enumerate(state_rows):
    n = r["life"]
    for c, s in enumerate(r["states"]):
        ax.barh(i, 1, left=-(n - 1 - c), color=cols[s], height=0.8, linewidth=0)
ax.set_yticks(range(len(state_rows)), [f"Engine {r['engine']}" for r in state_rows], fontsize=7)
ax.axvline(-full["R_acc"], color="black", lw=0.8, ls="--")
ax.set(title="Validation engines: proposed states over life (x = −true RUL; failure at 0)", xlabel="−True RUL (cycles)")
ax.grid(False)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=col, label=s) for s, col in cols.items()] +
          [plt.Line2D([], [], color="black", ls="--", lw=0.8, label=f"R_acc = {full['R_acc']:.0f}")],
          fontsize=7, loc="lower left", ncol=6, bbox_to_anchor=(0, -0.28))
save_fig(fig, "13_validation_state_timelines.png")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
leads = [life[u] - 1 - int(np.flatnonzero(val_tables[u]["pred"] <= full["R_acc"])[0]) for u in val_units
         if (val_tables[u]["pred"] <= full["R_acc"]).any()]
ax.hist(leads, bins=12, color=C_BLUE, edgecolor="white")
ax.axvline(full["R_acc"], color="black", ls="--", lw=1, label=f"R_acc = {full['R_acc']:.0f}")
ax.axvspan(acc_q10, acc_q90, color=C_ORANGE, alpha=0.15, label="Training engines' acceleration RUL q10–q90")
ax.set(title="LSTM on validation engines: true RUL at first Near-Failure flag", xlabel="True RUL at first flag (cycles)", ylabel="Engines")
ax.legend(fontsize=7)
save_fig(fig, "14_validation_near_failure_lead.png")
print(json.dumps({k: results[k] for k in ("full_80", "folds", "bootstrap_ci95", "heldout_summary")}, indent=1, default=float)[:7000])
print(json.dumps(results["near_failure_validation_engines"]["lstm_full_R"], indent=1, default=float))
print(json.dumps(results["near_failure_validation_engines"]["datf_full_R"], indent=1, default=float)[:1500])
print("fold-R LSTM lead medians:", [(x["R_acc"], x["true_rul_at_first_flag"].get("median"), x["missed_engines"],
                                    x["early_flags_true_rul_gt_train_q90"], x["late_flags_true_rul_lt_train_q10"]) for x in nf["lstm_by_fold_R"]])
print(json.dumps(results["combined_states_validation"], indent=1, default=float)[:3000])
