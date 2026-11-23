"""Health-state threshold analysis for NASA C-MAPSS FD001 (ANALYSIS ONLY - writes only to this directory).

Run from the project root:  .\\.venv\\Scripts\\python.exe outputs\\health_threshold_analysis\\reproduce_analysis.py

Data use: threshold discovery on the 80 TRAINING engines only; the 20 VALIDATION engines (held out when the models
were trained) are used only to check how frozen rules behave; the official TEST set is never loaded.
Reads (never writes): data/raw/train_FD001.txt, data/processed/preprocessor.joblib (saved scaler, split),
models/checkpoints/datf/final/datf_full_seed*.pt (DATF-Net inference only; no training).
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
RNG = np.random.default_rng(0)
BASE = 10          # baseline cycles (same definition as DATF-Net)
W = 10             # causal smoothing window (cycles) for the degradation index
K_SPC = 3.0        # control-limit multiplier (Shewhart convention)
PERSIST = 10       # cycles a crossing must persist to count as a transition
Q_CRIT = 0.10      # failure-level quantile for the near-failure boundary (sensitivity: 0.05, 0.20)
C_BLUE, C_ORANGE, C_AQUA, C_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8980"
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8980",
                     "axes.titlesize": 10, "legend.frameon": False})
results: dict = {"protocol": {"discovery_engines": "80 training engines (baseline split)",
                              "check_engines": "20 validation engines (held out)", "test_set": "not loaded",
                              "baseline_cycles": BASE, "smoothing_window": W, "spc_k": K_SPC,
                              "persistence": PERSIST, "critical_quantile": Q_CRIT}}


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def causal_mean(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def causal_slope(x, w):
    out = np.full(len(x), np.nan)
    tt = np.arange(w) - (w - 1) / 2
    for t in range(w - 1, len(x)):
        out[t] = (tt * x[t - w + 1:t + 1]).sum() / (tt ** 2).sum()
    return out


def first_sustained(mask, persist):
    """First index i such that mask[i:i+persist] are all True (or remaining cycles, if fewer)."""
    n = len(mask)
    for i in range(n):
        j = min(n, i + persist)
        if mask[i:j].all():
            return i
    return None


# ================= data (training + validation engines only) =================
prep = Preprocessor.load()
raw = load_train()
val_units = sorted(int(u) for u in prep.val_units)
all_units = sorted(int(u) for u in np.unique(raw["unit"]))
train_units = [u for u in all_units if u not in set(val_units)]
sens = [c for c in prep.feature_cols if c in SENSOR_COLS]
X = matrix(prep.transform(raw), sens)                    # training-scaler units (sigma of training data)
eng = {u: idx[np.argsort(raw["cycle"][idx])] for u, idx in groups(raw)}
dev = {u: X[eng[u]] - X[eng[u]][:BASE].mean(0) for u in all_units}   # deviation from own first-10-cycle baseline
life = {u: len(eng[u]) for u in all_units}
rul = {u: np.arange(life[u])[::-1] for u in all_units}             # true, uncapped (training labels)

# ================= STEP 2: data findings =================
L = np.array([life[u] for u in train_units])
results["data"] = {"n_train_engines": len(train_units), "n_val_engines": len(val_units),
                   "life": {"min": int(L.min()), "q25": float(np.percentile(L, 25)), "median": float(np.median(L)),
                            "q75": float(np.percentile(L, 75)), "max": int(L.max()), "mean": float(L.mean()),
                            "sd": float(L.std(ddof=1))},
                   "frac_cycles_at_cap_125": float(np.mean(np.concatenate([rul[u] >= 125 for u in train_units])))}
sens_stats = []
for j, c in enumerate(sens):
    r = np.array([np.corrcoef(np.arange(life[u]), X[eng[u], j])[0, 1] if X[eng[u], j].std() > 0 else np.nan
                  for u in train_units])
    med = float(np.nanmedian(r))
    early_sd = float(np.concatenate([dev[u][:BASE, j] for u in train_units]).std())
    late_mean = float(np.concatenate([dev[u][-BASE:, j] for u in train_units]).mean())
    sens_stats.append({"sensor": c, "median_r_with_cycle": med,
                       "frac_engines_consistent_trend": float(np.nanmean((np.abs(r) > 0.5) & (np.sign(r) == np.sign(med)))),
                       "late_mean_dev_sigma": late_mean, "early_noise_sd_sigma": early_sd,
                       "snr_late_vs_early": abs(late_mean) / early_sd})
results["sensors"] = sens_stats

# ================= unsupervised degradation index (DI): PC1 of baseline deviations =================
P = np.concatenate([dev[u] for u in train_units])
evals, evecs = np.linalg.eigh(np.cov(P, rowvar=False))
o = np.argsort(evals)[::-1]
evals, evecs = evals[o], evecs[:, o]
pc1 = evecs[:, 0]
if np.corrcoef(P @ pc1, np.concatenate([np.arange(life[u]) for u in train_units]))[0, 1] < 0:
    pc1 = -pc1  # orientation from elapsed cycles only (no RUL, no model)
DI = {u: dev[u] @ pc1 for u in all_units}
DIs = {u: causal_mean(DI[u], W) for u in all_units}
DIslope = {u: causal_slope(DI[u], W) for u in all_units}
sigma0 = float(np.concatenate([DI[u][:BASE] - DI[u][:BASE].mean() for u in train_units]).std(ddof=1))
early_s = np.concatenate([DIs[u][BASE:30] for u in train_units])
sigma_w = float(np.concatenate([DIs[u][W - 1:BASE + 20] - DI[u][:BASE].mean() for u in train_units]).std(ddof=1))
results["degradation_index"] = {
    "definition": "DI_t = w . (x_t - b_e); x_t scaled sensors (15), b_e mean of the engine's first 10 cycles, "
                  "w = first principal component of pooled training deviations (unsupervised; sign from elapsed cycles)",
    "explained_variance_ratio_pc1_to_pc4": [float(v) for v in (evals[:4] / evals.sum())],
    "weights": {c: float(v) for c, v in zip(sens, pc1)},
    "raw_noise_sd_first10": sigma0, "smoothed_noise_sd_cycles_10_to_30": sigma_w}

# ================= STEP 4B: change points per engine =================
def fit_flat(y):
    return float(((y - y.mean()) ** 2).sum())


def fit_hinge(y, power):
    t = np.arange(len(y), dtype=float)
    best = (np.inf, None)
    for tau in range(BASE, len(y) - 10):
        h = np.maximum(0.0, t - tau) ** power
        A = np.column_stack([np.ones_like(t), h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((y - A @ coef) ** 2).sum())
        if sse < best[0]:
            best = (sse, tau)
    return best


def fit_two_break(y, step=3):
    """flat -> slope b1 after tau1 -> additional slope b2 after tau2 (continuous piecewise linear)."""
    t = np.arange(len(y), dtype=float)
    n = len(y)
    best = (np.inf, None, None)
    grid = np.arange(BASE, n - 10, step)
    for t1 in grid:
        h1 = np.maximum(0.0, t - t1)
        t2s = grid[grid > t1 + 5]
        if len(t2s) == 0:
            continue
        H2 = np.maximum(0.0, t[:, None] - t2s[None, :])                      # (n, m)
        ones = np.ones(n)
        # normal equations for each t2 (3x3), batched
        G = np.empty((len(t2s), 3, 3))
        r = np.empty((len(t2s), 3))
        G[:, 0, 0] = n; G[:, 0, 1] = G[:, 1, 0] = h1.sum(); G[:, 1, 1] = (h1 * h1).sum()
        G[:, 0, 2] = G[:, 2, 0] = H2.sum(0); G[:, 1, 2] = G[:, 2, 1] = (h1[:, None] * H2).sum(0)
        G[:, 2, 2] = (H2 * H2).sum(0)
        r[:, 0] = y.sum(); r[:, 1] = (h1 * y).sum(); r[:, 2] = (H2 * y[:, None]).sum(0)
        coef = np.linalg.solve(G, r[..., None])[..., 0]
        pred = coef[:, 0][None, :] * ones[:, None] + coef[:, 1][None, :] * h1[:, None] + coef[:, 2][None, :] * H2
        sse = ((y[:, None] - pred) ** 2).sum(0)
        k = int(np.argmin(sse))
        if sse[k] < best[0]:
            best = (float(sse[k]), int(t1), int(t2s[k]))
    return best


def bic(sse, n, k):
    return n * np.log(sse / n) + k * np.log(n)


rows = []
L1 = K_SPC * sigma_w                                                       # Healthy -> Degrading control limit
for u in all_units:
    y = DI[u]
    n = len(y)
    s_flat = fit_flat(y)
    s_lin, tau_lin = fit_hinge(y, 1)
    s_quad, tau_quad = fit_hinge(y, 2)
    s_two, t1, t2 = fit_two_break(y)
    bics = {"flat": bic(s_flat, n, 2), "hinge_linear": bic(s_lin, n, 4), "hinge_quadratic": bic(s_quad, n, 4),
            "two_breakpoints": bic(s_two, n, 6)}
    spc = first_sustained(DIs[u][BASE:] > L1, PERSIST)
    spc_idx = None if spc is None else spc + BASE
    rows.append({"engine": u, "split": "train" if u in train_units else "validation", "life": n,
                 "tau_hinge_linear": tau_lin, "rul_at_onset_hinge_linear": n - 1 - tau_lin,
                 "tau_hinge_quadratic": tau_quad, "rul_at_onset_hinge_quadratic": n - 1 - tau_quad,
                 "tau1_two_break": t1, "tau2_two_break": t2, "rul_at_tau1": n - 1 - t1, "rul_at_tau2": n - 1 - t2,
                 "best_model_bic": min(bics, key=bics.get),
                 "dbic_two_break_vs_hinge_quadratic": float(bics["two_breakpoints"] - bics["hinge_quadratic"]),
                 "dbic_hinge_quadratic_vs_flat": float(bics["hinge_quadratic"] - bics["flat"]),
                 "spc_onset_cycle": spc_idx, "rul_at_spc_onset": None if spc_idx is None else n - 1 - spc_idx,
                 "frac_life_at_spc_onset": None if spc_idx is None else spc_idx / n,
                 "di_smoothed_at_failure": float(DIs[u][-1])})

tr_rows = [r for r in rows if r["split"] == "train"]


def dist(v):
    v = np.array([x for x in v if x is not None], dtype=float)
    return {"n": int(len(v)), "min": float(v.min()), "q10": float(np.percentile(v, 10)), "q25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q75": float(np.percentile(v, 75)), "q90": float(np.percentile(v, 90)),
            "max": float(v.max()), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


model_counts = {m: sum(r["best_model_bic"] == m for r in tr_rows) for m in
                ("flat", "hinge_linear", "hinge_quadratic", "two_breakpoints")}
results["change_points_train"] = {
    "best_model_by_bic_counts": model_counts,
    "engines_where_change_detected_dbic_lt_minus10": int(sum(r["dbic_hinge_quadratic_vs_flat"] < -10 for r in tr_rows)),
    "rul_at_onset_hinge_quadratic": dist([r["rul_at_onset_hinge_quadratic"] for r in tr_rows]),
    "rul_at_onset_hinge_linear": dist([r["rul_at_onset_hinge_linear"] for r in tr_rows]),
    "frac_life_at_onset_hinge_quadratic": dist([r["tau_hinge_quadratic"] / r["life"] for r in tr_rows]),
    "two_break_rul_at_tau1": dist([r["rul_at_tau1"] for r in tr_rows]),
    "two_break_rul_at_tau2": dist([r["rul_at_tau2"] for r in tr_rows]),
    "two_break_better_than_quadratic_dbic_lt_minus10": int(sum(r["dbic_two_break_vs_hinge_quadratic"] < -10 for r in tr_rows)),
    "spc_limit_L1": L1, "rul_at_spc_onset": dist([r["rul_at_spc_onset"] for r in tr_rows]),
    "frac_life_at_spc_onset": dist([r["frac_life_at_spc_onset"] for r in tr_rows]),
    "engines_without_spc_onset": int(sum(r["rul_at_spc_onset"] is None for r in tr_rows)),
}

# ================= STEP 4D: RUL-conditioned degradation (pooled curve + engine bootstrap) =================
RMAX = 250


def pooled_curve(units):
    s = np.zeros(RMAX + 1); c = np.zeros(RMAX + 1); s2 = np.zeros(RMAX + 1)
    for u in units:
        r = rul[u]; m = r <= RMAX
        np.add.at(s, r[m], DIs[u][m]); np.add.at(s2, r[m], DIs[u][m] ** 2); np.add.at(c, r[m], 1)
    mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    sd = np.sqrt(np.maximum(np.where(c > 0, s2 / np.maximum(c, 1), np.nan) - mean ** 2, 0))
    return mean, sd, c


def knee(mean, cnt, min_eng=20):
    """Single breakpoint of mean DI vs RUL: flat for RUL >= r*, quadratic rise below (fit where >= min_eng engines)."""
    r = np.arange(RMAX + 1)
    ok = cnt >= min_eng
    rr, yy = r[ok].astype(float), mean[ok]
    best = (np.inf, None)
    for rs in range(10, int(rr.max()) - 10):
        h = np.maximum(0.0, rs - rr) ** 2
        A = np.column_stack([np.ones_like(rr), h])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        sse = float(((yy - A @ coef) ** 2).sum())
        if sse < best[0]:
            best = (sse, rs)
    return best[1]


mean_c, sd_c, cnt_c = pooled_curve(train_units)
k_point = knee(mean_c, cnt_c)
boot = []
for _ in range(500):
    samp = list(RNG.choice(train_units, len(train_units), replace=True))
    m_b, _, c_b = pooled_curve(samp)
    boot.append(knee(m_b, c_b))
boot = np.array(boot)
slope_by_rul = np.full(RMAX + 1, np.nan)
for r_ in range(RMAX + 1):
    v = [DIslope[u][rul[u] == r_][0] for u in train_units if (rul[u] == r_).any() and not np.isnan(DIslope[u][rul[u] == r_][0])]
    if len(v) >= 20:
        slope_by_rul[r_] = np.mean(v)
results["rul_conditioned_train"] = {
    "knee_rul_pooled_mean_curve": int(k_point), "knee_bootstrap_500_engine_resamples": dist(boot),
    "knee_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "note": "knee of the POOLED mean curve; compare with the engine-level onset distribution for consistency"}
with open(OUT / "rul_conditioned_degradation.csv", "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["true_rul", "n_engines", "mean_DI_smoothed", "sd_DI_smoothed", "mean_DI_slope_per_cycle"])
    for r_ in range(RMAX + 1):
        wtr.writerow([r_, int(cnt_c[r_]), mean_c[r_], sd_c[r_], slope_by_rul[r_]])

# ================= STEP 4C: clustering (GMM, numpy EM, BIC) =================
def gmm_fit(Z, k, restarts=4, iters=200):
    n, d = Z.shape
    best = None
    for rs in range(restarts):
        rng = np.random.default_rng(100 + rs)
        mu = Z[rng.choice(n, k, replace=False)]
        cov = np.array([np.cov(Z, rowvar=False) + 1e-6 * np.eye(d)] * k)
        pi = np.full(k, 1 / k)
        ll_old = -np.inf
        for _ in range(iters):
            logp = np.empty((n, k))
            for j in range(k):
                diff = Z - mu[j]
                inv = np.linalg.inv(cov[j]); _, logdet = np.linalg.slogdet(cov[j])
                logp[:, j] = np.log(pi[j]) - 0.5 * (np.einsum("ni,ij,nj->n", diff, inv, diff) + logdet + d * np.log(2 * np.pi))
            mx = logp.max(1, keepdims=True)
            ll = float((mx[:, 0] + np.log(np.exp(logp - mx).sum(1))).sum())
            resp = np.exp(logp - mx); resp /= resp.sum(1, keepdims=True)
            nk = resp.sum(0) + 1e-9
            pi = nk / n
            mu = (resp.T @ Z) / nk[:, None]
            for j in range(k):
                diff = Z - mu[j]
                cov[j] = (resp[:, j, None] * diff).T @ diff / nk[j] + 1e-6 * np.eye(d)
            if abs(ll - ll_old) < 1e-4 * abs(ll):
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, mu.copy(), cov.copy(), pi.copy(), resp.argmax(1))
    ll, mu, cov, pi, lab = best
    p = k * (d + d * (d + 1) / 2) + (k - 1)
    return {"ll": ll, "bic": -2 * ll + p * np.log(n), "mu": mu, "labels": lab}


feat_units, feat_rul, Zlist = [], [], []
for u in train_units:
    ok = ~np.isnan(DIslope[u])
    Zlist.append(np.column_stack([DIs[u][ok], DIslope[u][ok]]))
    feat_units += [u] * int(ok.sum()); feat_rul += list(rul[u][ok])
Z = np.concatenate(Zlist)
Zs = (Z - Z.mean(0)) / Z.std(0)
feat_units, feat_rul = np.array(feat_units), np.array(feat_rul)
gmm = {k: gmm_fit(Zs, k) for k in range(1, 7)}
bic_k = {k: float(g["bic"]) for k, g in gmm.items()}
g3 = gmm[3]
order3 = np.argsort(g3["mu"][:, 0])                       # clusters ordered by mean DI level
lab3 = np.argsort(order3)[g3["labels"]]
rul_by_cluster = {int(c): dist(feat_rul[lab3 == c]) for c in range(3)}
monotone = 0
for u in train_units:
    s = lab3[feat_units == u]
    monotone += int((np.diff(s) >= 0).all())
switches = [int((np.diff(lab3[feat_units == u]) != 0).sum()) for u in train_units]
results["clustering_train"] = {
    "features": "per cycle: smoothed DI level and 10-cycle DI slope (standardised)",
    "bic_by_k": bic_k, "k_min_bic": int(min(bic_k, key=bic_k.get)),
    "k3_rul_by_cluster_ordered_by_DI": rul_by_cluster,
    "k3_engines_with_monotone_cluster_sequence": int(monotone),
    "k3_label_switches_per_engine": dist(switches)}

# ================= STEP 3: DATF-Net health index (inference only) =================
hi_res = {"available": False}
try:
    import torch
    from src.data.degradation import load_train_val
    from src.train_datf import load_checkpoint, predict
    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trw, vaw = load_train_val(prep)
    his = {"train": [], "validation": []}
    ckpts = sorted((ROOT / "models" / "checkpoints" / "datf" / "final").glob("datf_full_seed*.pt"))
    for ck in ckpts:
        m, _ = load_checkpoint(ck, len(prep.feature_cols), prep.seq_len, dev_t)
        for name, d in (("train", trw), ("validation", vaw)):
            _, hi = predict(m, d["X"], d["b"], float(prep.rul_clip), dev_t)
            his[name].append(hi[:, -1])
    hi_res = {"available": True, "checkpoints": [c.name for c in ckpts], "note":
              "HI_T = health index at the last cycle of each window; mean over the 5 seeds. Training engines are "
              "IN-SAMPLE for DATF-Net; validation engines are held out."}
    for name, d in (("train", trw), ("validation", vaw)):
        h = np.mean(his[name], axis=0)
        y = d["y"]  # clipped RUL at window end
        units, ends = d["unit"], d["end_cycle"]
        true_unclipped = np.array([life[int(u)] - int(e) for u, e in zip(units, ends)])
        steps, inc = [], []
        for u in np.unique(units):
            hu = h[units == u][np.argsort(ends[units == u])]
            dd = np.diff(hu); steps.append(dd); inc.append(np.mean(dd > 0))
        steps = np.concatenate(steps)
        hi_res[name] = {
            "n_windows": int(len(h)), "corr_hi_vs_clipped_rul": float(np.corrcoef(h, y)[0, 1]),
            "residual_sd_vs_target": float(np.std(h - y / prep.rul_clip)),
            "hi_when_true_rul_ge_125": dist(h[true_unclipped >= 125]),
            "hi_when_true_rul_le_10": dist(h[true_unclipped <= 10]),
            "hi_when_true_rul_26_to_50": dist(h[(true_unclipped > 25) & (true_unclipped <= 50)]),
            "frac_steps_increasing_per_engine": dist(inc),
            "step_abs_mean": float(np.abs(steps).mean()), "true_per_cycle_decrease": 1 / prep.rul_clip,
            "hist_counts_10_bins_0_1": np.histogram(h, bins=10, range=(0, 1))[0].tolist()}
        if name == "validation":
            hi_val = (h, true_unclipped, units)
        else:
            hi_tr = (h, true_unclipped, units)
except Exception as e:  # report instead of failing the whole analysis
    hi_res = {"available": False, "error": repr(e)}
results["datf_hi"] = hi_res

# ================= candidate rule (frozen from training) and its behaviour =================
fail_levels = np.array([r["di_smoothed_at_failure"] for r in tr_rows])
L2 = {q: float(np.quantile(fail_levels, q)) for q in (0.05, 0.10, 0.20)}


def apply_rule(u, l1, l2, persist=PERSIST):
    """Latched states from the smoothed DI: 0 healthy, 1 degrading (sustained > l1), 2 near failure (sustained > l2)."""
    y = DIs[u]
    d1 = first_sustained(y[BASE:] > l1, persist)
    d2 = first_sustained(y[BASE:] > l2, persist)
    d1 = None if d1 is None else d1 + BASE
    d2 = None if d2 is None else d2 + BASE
    n = life[u]
    return {"rul_at_degrading": None if d1 is None else n - 1 - d1, "rul_at_near_failure": None if d2 is None else n - 1 - d2,
            "reached_near_failure_before_end": d2 is not None,
            "unlatched_dips_after_near_failure": None if d2 is None else float(np.mean(y[d2:] <= l2))}


rule_eval = {}
for q, l2 in L2.items():
    for split, units in (("train", train_units), ("validation", val_units)):
        ev = [apply_rule(u, L1, l2) for u in units]
        rule_eval[f"q{q}_{split}"] = {
            "L2": l2, "engines": len(units),
            "rul_at_degrading": dist([e["rul_at_degrading"] for e in ev]),
            "rul_at_near_failure": dist([e["rul_at_near_failure"] for e in ev]) if any(e["rul_at_near_failure"] is not None for e in ev) else None,
            "engines_never_flagged_near_failure": int(sum(not e["reached_near_failure_before_end"] for e in ev)),
            "mean_frac_cycles_below_L2_after_entry": float(np.mean([e["unlatched_dips_after_near_failure"] for e in ev
                                                                     if e["unlatched_dips_after_near_failure"] is not None]))}
# sensitivity of the Degrading onset to k, window and persistence (training engines)
sens_rows = []
for k in (2.5, 3.0, 3.5):
    for w in (5, 10, 20):
        sw = float(np.concatenate([causal_mean(DI[u], w)[w - 1:BASE + 20] - DI[u][:BASE].mean() for u in train_units]).std(ddof=1))
        for pers in (5, 10):
            on = []
            for u in train_units:
                ys = causal_mean(DI[u], w)
                i = first_sustained(ys[BASE:] > k * sw, pers)
                on.append(None if i is None else life[u] - 1 - (i + BASE))
            dd = dist(on)
            sens_rows.append({"k": k, "window": w, "persist": pers, "limit": k * sw, "median_rul_at_onset": dd["median"],
                              "q10": dd["q10"], "q90": dd["q90"], "n_detected": dd["n"]})
results["candidate_rule"] = {"L1_degrading_limit": L1, "L2_near_failure_limits_by_quantile": L2, "evaluation": rule_eval,
                             "onset_sensitivity_train": sens_rows}

# ================= outputs: CSV, JSON =================
with open(OUT / "engine_level_summary.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wtr.writeheader(); wtr.writerows(rows)
with open(OUT / "sensor_trend_summary.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(sens_stats[0].keys()))
    wtr.writeheader(); wtr.writerows(sens_stats)
(OUT / "analysis_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

# ================= plots =================
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(L, bins=20, color=C_BLUE, edgecolor="white")
ax.set(title="Training-engine lifetimes (80 engines)", xlabel="Cycles to failure", ylabel="Engines")
save_fig(fig, "01_lifetimes.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
for u in train_units:
    ax.plot(rul[u], DIs[u], color=C_GREY, lw=0.5, alpha=0.5)
ok = cnt_c >= 20
ax.plot(np.arange(RMAX + 1)[ok], mean_c[ok], color=C_BLUE, lw=2, label="Mean over engines (≥20 engines)")
ax.axhline(L1, color=S_WARN, ls="--", lw=1, label=f"Baseline control limit {K_SPC:.0f}σ ({L1:.2f})")
ax.axvline(125, color=C_GREY, ls=":", lw=1, label="RUL cap used in training (125)")
ax.set_xlim(RMAX, 0)
ax.set(title="Sensor-derived degradation index vs true RUL (training engines)", xlabel="True RUL (cycles, failure at 0)",
       ylabel="DI (smoothed, training-σ units)")
ax.legend(fontsize=8, loc="upper left")
save_fig(fig, "02_degradation_index_vs_rul.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
for j, c in enumerate(sens):
    s = np.zeros(RMAX + 1); cc = np.zeros(RMAX + 1)
    for u in train_units:
        m = rul[u] <= RMAX
        np.add.at(s, rul[u][m], causal_mean(dev[u][:, j], W)[m] * np.sign(sens_stats[j]["median_r_with_cycle"] or 1)); np.add.at(cc, rul[u][m], 1)
    okc = cc >= 20
    ax.plot(np.arange(RMAX + 1)[okc], (s / np.maximum(cc, 1))[okc], lw=1, color=C_BLUE if sens_stats[j]["frac_engines_consistent_trend"] >= 0.95 else C_ORANGE,
            alpha=0.8)
ax.plot([], [], color=C_BLUE, label="Consistent sensors (same trend in ≥95% of engines)")
ax.plot([], [], color=C_ORANGE, label="Inconsistent sensors")
ax.set_xlim(RMAX, 0)
ax.set(title="Per-sensor deviation from own baseline (sign-aligned), mean over training engines",
       xlabel="True RUL (cycles)", ylabel="Deviation (training σ)")
ax.legend(fontsize=8)
save_fig(fig, "03_sensor_deviation_vs_rul.png")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].hist([r["rul_at_onset_hinge_quadratic"] for r in tr_rows], bins=20, color=C_BLUE, edgecolor="white", alpha=0.9,
            label="Change point (flat→quadratic)")
axs[0].hist([r["rul_at_spc_onset"] for r in tr_rows if r["rul_at_spc_onset"] is not None], bins=20, color=C_ORANGE,
            edgecolor="white", alpha=0.6, label=f"Control-limit onset ({K_SPC:.0f}σ, {PERSIST} cycles)")
axs[0].set(title="RUL at degradation onset, per engine", xlabel="RUL at onset (cycles)", ylabel="Engines")
axs[0].legend(fontsize=7)
axs[1].hist([r["tau_hinge_quadratic"] / r["life"] for r in tr_rows], bins=20, color=C_BLUE, edgecolor="white")
axs[1].set(title="Onset as fraction of engine life", xlabel="Onset cycle / life", ylabel="Engines")
save_fig(fig, "04_change_point_onsets.png")

ex = sorted(tr_rows, key=lambda r: r["life"])
ex = [ex[0], ex[len(ex) // 3], ex[2 * len(ex) // 3], ex[-1]]
fig, axs = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
for ax, r in zip(axs, ex):
    u = r["engine"]; t = np.arange(life[u])
    ax.plot(t, DI[u], color=C_GREY, lw=0.6, label="DI (raw)")
    ax.plot(t, DIs[u], color=C_BLUE, lw=1.2, label="DI (smoothed)")
    ax.axvline(r["tau_hinge_quadratic"], color=C_ORANGE, ls="--", lw=1, label="Change point")
    if r["spc_onset_cycle"] is not None:
        ax.axvline(r["spc_onset_cycle"], color=S_WARN, ls=":", lw=1.2, label="Control-limit onset")
    ax.set(title=f"Engine {u} (life {r['life']})", xlabel="Cycle")
axs[0].set_ylabel("DI (σ)"); axs[0].legend(fontsize=6)
save_fig(fig, "05_change_point_examples.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].plot(np.arange(RMAX + 1)[ok], mean_c[ok], color=C_BLUE, lw=2, label="Mean DI")
axs[0].fill_between(np.arange(RMAX + 1)[ok], (mean_c - sd_c)[ok], (mean_c + sd_c)[ok], color=C_BLUE, alpha=0.15, label="±1 SD across engines")
axs[0].axvspan(np.percentile(boot, 2.5), np.percentile(boot, 97.5), color=C_ORANGE, alpha=0.2, label="Knee 95% bootstrap CI")
axs[0].axvline(k_point, color=C_ORANGE, lw=1.2, label=f"Knee (pooled) = {k_point}")
axs[0].set_xlim(RMAX, 0); axs[0].set(title="RUL-conditioned degradation level", xlabel="True RUL", ylabel="DI (σ)")
axs[0].legend(fontsize=7)
oks = ~np.isnan(slope_by_rul)
axs[1].plot(np.arange(RMAX + 1)[oks], slope_by_rul[oks], color=C_BLUE, lw=1.5)
axs[1].set_xlim(RMAX, 0); axs[1].set(title="Mean DI slope vs RUL (rate of degradation)", xlabel="True RUL", ylabel="DI slope (σ/cycle)")
save_fig(fig, "06_rul_conditioned.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].plot(list(bic_k), list(bic_k.values()), marker="o", color=C_BLUE)
axs[0].set(title="GMM model selection (lower BIC = better)", xlabel="Number of clusters k", ylabel="BIC")
sub = RNG.choice(len(Z), min(4000, len(Z)), replace=False)
for c, col in zip(range(3), (S_GOOD, S_WARN, S_CRIT)):
    m = lab3[sub] == c
    axs[1].scatter(feat_rul[sub][m], Z[sub][m, 0], s=3, color=col, alpha=0.5, label=f"Cluster {c + 1}")
axs[1].set_xlim(RMAX, 0); axs[1].set(title="k=3 clusters (DI level + slope) vs true RUL", xlabel="True RUL", ylabel="DI (σ)")
axs[1].legend(fontsize=7, markerscale=3)
save_fig(fig, "07_clustering.png")

if hi_res.get("available"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (h, tr_, uu), title in ((axs[0], hi_tr, "Training engines (in-sample)"), (axs[1], hi_val, "Validation engines (held out)")):
        ax.scatter(np.minimum(tr_, RMAX), h, s=2, color=C_BLUE, alpha=0.25)
        r_ = np.arange(RMAX + 1)
        ax.plot(r_, np.minimum(r_, 125) / 125, color=C_ORANGE, lw=1.5, label="Training target min(RUL,125)/125")
        ax.set_xlim(RMAX, 0); ax.set(title=f"DATF-Net HI vs true RUL – {title}", xlabel="True RUL (capped at 250 for display)")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("HI at window end (mean of 5 seeds)")
    save_fig(fig, "08_datf_hi_vs_rul.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
l2 = L2[Q_CRIT]
for u in val_units:
    ax.plot(rul[u], DIs[u], color=C_GREY, lw=0.7, alpha=0.7)
ax.axhspan(-10, L1, color=S_GOOD, alpha=0.08); ax.axhspan(L1, l2, color=S_WARN, alpha=0.10); ax.axhspan(l2, 20, color=S_CRIT, alpha=0.08)
ax.axhline(L1, color=S_WARN, ls="--", lw=1, label=f"L1 Healthy→Degrading = {L1:.2f} (3σ baseline noise)")
ax.axhline(l2, color=S_CRIT, ls="--", lw=1, label=f"L2 Degrading→Near failure = {l2:.2f} (q{Q_CRIT:.2f} of failure-time DI)")
ax.set_xlim(RMAX, 0); ax.set_ylim(min(-1.5, np.nanmin([DIs[u].min() for u in val_units])), max(np.nanmax([DIs[u].max() for u in val_units]) + 0.5, l2 + 1))
ax.set(title="Candidate rule (frozen on training engines) applied to the 20 validation engines",
       xlabel="True RUL (cycles)", ylabel="DI (smoothed, σ)")
ax.legend(fontsize=7, loc="upper left")
save_fig(fig, "09_candidate_rule_validation.png")
print(json.dumps({k: results[k] for k in ("data", "degradation_index", "change_points_train", "rul_conditioned_train",
                                          "clustering_train")}, indent=1, default=float)[:6000])
print(json.dumps(results["datf_hi"], indent=1, default=float)[:3500])
print(json.dumps({k: v for k, v in results["candidate_rule"].items() if k != "onset_sensitivity_train"}, indent=1, default=float)[:4000])
print("sensitivity:", [(r["k"], r["window"], r["persist"], r["median_rul_at_onset"], r["n_detected"]) for r in sens_rows])
