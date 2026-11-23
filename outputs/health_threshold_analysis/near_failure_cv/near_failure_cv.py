"""5-fold ENGINE-level cross-validated Near-Failure evaluation (VALIDATION ONLY).

Run from the project root:
    .\\.venv\\Scripts\\python.exe outputs\\health_threshold_analysis\\near_failure_cv\\near_failure_cv.py

* Engines: the 100 engines of train_FD001.txt, split into 5 folds of 20. The official test files are never read.
* Per fold: the 80 non-held-out engines are split 64 inner-train / 16 inner-validation. A NEW scaler (in memory only)
  is fitted on the 64 inner-train engines; models are trained on them with early stopping on the 16 inner-validation
  engines; sensor parameters (DI weights, L1, R_acc) are re-derived on the 80 non-held-out engines; everything is
  evaluated on the 20 held-out engines.
* Models: LSTM baseline architecture (the deployed model type; primary, with MC-dropout uncertainty computed exactly
  as in production) and DATF-Net (secondary, point predictions), 5 seeds each, identical training conditions to the
  existing pipeline (src/train_datf.train_run).
* These are VALIDATION-ONLY models, saved under ./models/ next to this script. The production checkpoints in
  models/checkpoints/ and all existing results are never written.
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
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.degradation import build_windows, engine_baseline  # noqa: E402
from src.data.load_data import SENSOR_COLS, load_train  # noqa: E402
from src.data.preprocess import Preprocessor, add_rul, windows_for_engine  # noqa: E402
from src.data.table import groups, matrix, take  # noqa: E402
from src.train_datf import RunSpec, TrainSettings, load_checkpoint, predict, train_run  # noqa: E402

OUT = Path(__file__).resolve().parent
MODELS = OUT / "models"
SEEDS = [42, 43, 44, 45, 46]
N_FOLDS, FOLD_SEED = 5, 2025
BASE, W, K_SPC, M_PERSIST = 10, 10, 3.0, 20      # m = 20: implementation policy (see cv_validation_report.md)
CLIP, SEQ = 125, 30
MC_SAMPLES, MC_SEED = 30, 0                       # identical to production (src/config.py)
N_BOOT = 2000
GATES = (0, 10, 29)   # earliest cycle at which the latch may trigger: none / 10-cycle baseline (Insufficient history) / first full window
PRIMARY = "confident_g29"   # recommended after the gating comparison: latch outside the Boundary band, from the first full window
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_BLUE, C_ORANGE, C_GREY = "#2a78d6", "#eb6834", "#8a8980"
S_GOOD, S_WARN, S_CRIT, S_UNC = "#0ca30c", "#fab219", "#d03b3b", "#7f7f7f"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8980",
                     "axes.titlesize": 10, "legend.frameon": False})


def dist(v):
    v = np.array([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "min": float(v.min()), "q10": float(np.percentile(v, 10)), "q25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q75": float(np.percentile(v, 75)), "q90": float(np.percentile(v, 90)),
            "max": float(v.max()), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


# ---------------- sensor helpers (same definitions as cv_validation.py) ----------------
def causal_mean(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0)); t = np.arange(len(x)); lo = np.maximum(0, t - w + 1)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


def first_sustained(mask, persist):
    run = 0
    for i in range(len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= persist:
            return i - persist + 1
    return len(mask) - run if run > 0 else None


def fit_hinge_quad(y):
    t = np.arange(len(y), dtype=float); best = (np.inf, None)
    for tau in range(BASE, len(y) - 10):
        A = np.column_stack([np.ones_like(t), np.maximum(0.0, t - tau) ** 2])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((y - A @ coef) ** 2).sum())
        if sse < best[0]:
            best = (sse, tau)
    return best


def fit_two_break(y, step=3):
    t = np.arange(len(y), dtype=float); n = len(y); best = (np.inf, None, None); grid = np.arange(BASE, n - 10, step)
    for t1 in grid:
        h1 = np.maximum(0.0, t - t1); t2s = grid[grid > t1 + 5]
        if len(t2s) == 0:
            continue
        H2 = np.maximum(0.0, t[:, None] - t2s[None, :])
        G = np.empty((len(t2s), 3, 3)); r = np.empty((len(t2s), 3))
        G[:, 0, 0] = n; G[:, 0, 1] = G[:, 1, 0] = h1.sum(); G[:, 1, 1] = (h1 * h1).sum()
        G[:, 0, 2] = G[:, 2, 0] = H2.sum(0); G[:, 1, 2] = G[:, 2, 1] = (h1[:, None] * H2).sum(0); G[:, 2, 2] = (H2 * H2).sum(0)
        r[:, 0] = y.sum(); r[:, 1] = (h1 * y).sum(); r[:, 2] = (H2 * y[:, None]).sum(0)
        coef = np.linalg.solve(G, r[..., None])[..., 0]
        pred = coef[:, 0][None, :] + coef[:, 1][None, :] * h1[:, None] + coef[:, 2][None, :] * H2
        sse = ((y[:, None] - pred) ** 2).sum(0); k = int(np.argmin(sse))
        if sse[k] < best[0]:
            best = (float(sse[k]), int(t1), int(t2s[k]))
    return best


def bic(sse, n, k):
    return n * np.log(sse / n) + k * np.log(n)


# ---------------- MC-dropout uncertainty exactly as src/predict.Predictor._mc_dropout_std, batched ----------------
@torch.no_grad()
def lstm_predict_with_uncertainty(model, Wn: np.ndarray):
    x = torch.from_numpy(Wn).to(DEVICE)
    point = model(x).cpu().numpy() * CLIP
    H = model.lstm(x)[0][:, -1]                                       # (B, hidden)
    gen = torch.Generator(device=DEVICE).manual_seed(MC_SEED)
    h = H[:, None, :].expand(-1, MC_SAMPLES, -1)                      # (B, n, hidden)
    for layer in model.head:                                          # same mask draw order/shapes as production
        if isinstance(layer, torch.nn.Dropout):
            keep = 1.0 - layer.p
            mask = torch.bernoulli(torch.full((MC_SAMPLES, h.shape[-1]), keep, device=DEVICE), generator=gen) / keep
            h = h * mask[None]
        else:
            h = layer(h)
    std = h.squeeze(-1).std(dim=1).cpu().numpy() * CLIP
    return np.clip(point, 0, None), std


# ================= data and folds =================
raw = load_train()
df = add_rul(raw, clip=CLIP)
units_all = sorted(int(u) for u in np.unique(raw["unit"]))
life = {u: int((raw["unit"] == u).sum()) for u in units_all}
perm = np.random.default_rng(FOLD_SEED).permutation(units_all)
folds = [sorted(int(u) for u in perm[i::N_FOLDS]) for i in range(N_FOLDS)]
MODELS.mkdir(parents=True, exist_ok=True)

engine_records = []     # one row per (fold, model, seed, held-out engine)
fold_meta = []
state_examples = {}
for k, held in enumerate(folds, start=1):
    rest = [u for u in units_all if u not in set(held)]
    inner_val = sorted(int(u) for u in np.random.default_rng(FOLD_SEED + k).choice(rest, 16, replace=False))
    inner_train = [u for u in rest if u not in set(inner_val)]
    prep = Preprocessor(SEQ, CLIP).fit(take(df, np.isin(df["unit"], inner_train)), inner_val)   # in memory only
    scaled = prep.transform(df)
    feats = prep.feature_cols
    tr = build_windows(scaled, feats, SEQ, CLIP, inner_train, BASE)
    va = build_windows(scaled, feats, SEQ, CLIP, inner_val, BASE)

    # ---- sensor parameters from the 80 non-held-out engines (fold scaler) ----
    sens = [c for c in feats if c in SENSOR_COLS]
    Xs = matrix(scaled, sens)
    eng = {u: idx[np.argsort(raw["cycle"][idx])] for u, idx in groups(raw)}
    dev = {u: Xs[eng[u]] - Xs[eng[u]][:BASE].mean(0) for u in units_all}
    P = np.concatenate([dev[u] for u in rest])
    ev, V = np.linalg.eigh(np.cov(P, rowvar=False)); w = V[:, np.argmax(ev)]
    if np.corrcoef(P @ w, np.concatenate([np.arange(life[u]) for u in rest]))[0, 1] < 0:
        w = -w
    DI = {u: dev[u] @ w for u in units_all}; DIs = {u: causal_mean(DI[u], W) for u in units_all}
    L1 = K_SPC * float(np.concatenate([DIs[u][W - 1:BASE + 20] - DI[u][:BASE].mean() for u in rest]).std(ddof=1))
    acc = []
    for u in rest:
        y = DI[u]; n = len(y)
        s_q, _ = fit_hinge_quad(y); s_2, _, t2 = fit_two_break(y)
        if bic(s_2, n, 6) - bic(s_q, n, 4) < -10:
            acc.append(n - 1 - t2)
    R = float(np.median(acc)); acc_q10, acc_q90 = float(np.percentile(acc, 10)), float(np.percentile(acc, 90))
    degr_cycle = {}
    for u in held:
        st = first_sustained(DIs[u][BASE:] > L1, M_PERSIST)
        degr_cycle[u] = None if st is None else min(st + BASE + M_PERSIST - 1, life[u] - 1)
    fold_meta.append({"fold": k, "held_out": held, "inner_val": inner_val, "n_inner_train": len(inner_train),
                      "n_features": len(feats), "L1": L1, "R_acc": R, "n_acc_supported": len(acc),
                      "acc_q10": acc_q10, "acc_q90": acc_q90})
    print(f"fold {k}: features {len(feats)}  L1 {L1:.3f}  R_acc {R:.1f} ({len(acc)}/80 supported)", flush=True)

    # held-out inputs (fold scaler)
    held_in = {}
    for u in held:
        Fe = matrix(scaled, feats)[eng[u]].astype(np.float32)
        Wp, _ = windows_for_engine(Fe, SEQ, pad=True)                 # LSTM: every cycle, as deployed
        Wf, ends = windows_for_engine(Fe, SEQ)                         # DATF-Net: full windows + baseline
        held_in[u] = (Wp, Wf, ends, np.repeat(engine_baseline(Fe, BASE)[None, :], len(ends), axis=0).astype(np.float32))

    for model_type, spec, settings in (("lstm", RunSpec("cv_lstm", "lstm"), TrainSettings()),
                                       ("datf", RunSpec("cv_datf", "datf", {}, 0.1, 0.1), TrainSettings(lambda_hi=0.1, lambda_mono=0.1))):
        for seed in SEEDS:
            ckpt = MODELS / f"fold{k}_{model_type}_seed{seed}.pt"
            hist_path = MODELS / f"fold{k}_{model_type}_seed{seed}_history.json"
            if not hist_path.exists():
                with open(MODELS / f"fold{k}_{model_type}_seed{seed}_epochs.log", "w") as fh:
                    hist = train_run(spec, tr, va, seed, settings, float(CLIP), ckpt, DEVICE,
                                     extra_meta={"validation_only": True, "fold": k, "feature_cols": feats,
                                                 "inner_train": inner_train, "inner_val": inner_val, "held_out": held},
                                     log=lambda s: fh.write(s + "\n"))
                hist_path.write_text(json.dumps(hist, indent=1, default=float))
            model, ck = load_checkpoint(ckpt, len(feats), SEQ, DEVICE)
            for u in held:
                Wp, Wf, ends, B = held_in[u]
                n = life[u]
                if model_type == "lstm":
                    p, unc = lstm_predict_with_uncertainty(model, Wp)
                    cyc = np.arange(n)
                else:
                    p, _ = predict(model, Wf, B, float(CLIP), DEVICE); unc = np.zeros_like(p); cyc = ends
                true = n - 1 - cyc
                truth_nf = true <= R
                simple = p <= R
                confident = (p + unc) <= R if model_type == "lstm" else simple
                alert = (p - unc) <= R if model_type == "lstm" else simple   # first entry into Boundary band or below
                boundary = np.abs(p - R) <= unc if model_type == "lstm" else np.zeros_like(simple)
                rec = {"fold": k, "model": model_type, "seed": seed, "engine": u, "life": n, "R_acc": R,
                       "acc_q10": acc_q10, "acc_q90": acc_q90, "best_epoch": ck["epoch"],
                       "degrading_declared_cycle": degr_cycle[u]}
                trig_list = []
                for gate in GATES:
                    allowed = cyc >= gate
                    trig_list += [(f"simple_g{gate}", simple & allowed), (f"confident_g{gate}", confident & allowed),
                                  (f"alert_g{gate}", alert & allowed)]
                for trig_name, trig in trig_list:
                    idx = np.flatnonzero(trig)
                    first = int(idx[0]) if len(idx) else None
                    latched = np.zeros(len(cyc), bool)
                    if first is not None:
                        latched[first:] = True
                    lead = None if first is None else int(true[first])
                    after = slice(first, None) if first is not None else slice(0, 0)
                    contradicted = (p[after] > R + unc[after]) if model_type == "lstm" else (p[after] > R)
                    rec.update({
                        f"{trig_name}_detected": first is not None, f"{trig_name}_lead": lead,
                        f"{trig_name}_early": bool(lead is not None and lead > acc_q90),
                        f"{trig_name}_late": bool(lead is None or lead < acc_q10),
                        f"{trig_name}_tp": int((latched & truth_nf).sum()), f"{trig_name}_fp": int((latched & ~truth_nf).sum()),
                        f"{trig_name}_fn": int((~latched & truth_nf).sum()),
                        f"{trig_name}_contradicted_cycles_after_latch": int(contradicted.sum()),
                        f"{trig_name}_max_pred_after_latch": None if first is None else float(p[after].max()),
                        f"{trig_name}_latched_while_sensor_healthy_cycles": int(sum(
                            1 for c_i, c in enumerate(cyc) if latched[c_i] and (degr_cycle[u] is None or c < degr_cycle[u]))),
                    })
                rec.update({"raw_tp": int((simple & truth_nf).sum()), "raw_fp": int((simple & ~truth_nf).sum()),
                            "raw_fn": int((~simple & truth_nf).sum()),
                            "raw_returns_above_boundary": int(np.sum(np.diff(simple.astype(int)) == -1)),
                            "boundary_cycles": int(boundary.sum()), "cycles": int(len(cyc)),
                            "abs_err_last_50_cycles": float(np.mean(np.abs(p[true <= 50] - true[true <= 50])))})
                engine_records.append(rec)
                if model_type == "lstm" and seed == SEEDS[0]:
                    state_examples[u] = {"cyc": cyc, "p": p, "unc": unc, "R": R, "degr": degr_cycle[u], "fold": k}
            print(f"  fold {k} {model_type} seed {seed}: best epoch {ck['epoch']}", flush=True)


# ================= aggregation + engine bootstrap CIs =================
def summarise(recs, trig):
    det = np.mean([r[f"{trig}_detected"] for r in recs])
    leads = [r[f"{trig}_lead"] for r in recs if r[f"{trig}_lead"] is not None]
    tp = sum(r[f"{trig}_tp"] for r in recs); fp = sum(r[f"{trig}_fp"] for r in recs); fn = sum(r[f"{trig}_fn"] for r in recs)
    return {"detection_rate": float(det), "lead": dist(leads), "median_lead": float(np.median(leads)) if leads else None,
            "early_rate": float(np.mean([r[f"{trig}_early"] for r in recs])),
            "late_or_missed_rate": float(np.mean([r[f"{trig}_late"] for r in recs])),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "engines_with_contradiction_after_latch": float(np.mean([r[f"{trig}_contradicted_cycles_after_latch"] > 0 for r in recs])),
            "contradicted_cycles_total": int(sum(r[f"{trig}_contradicted_cycles_after_latch"] for r in recs)),
            "latched_while_sensor_healthy_engines": float(np.mean([r[f"{trig}_latched_while_sensor_healthy_cycles"] > 0 for r in recs]))}


def boot_ci(recs, trig, n_boot=N_BOOT):
    rng = np.random.default_rng(11)
    engines = sorted({r["engine"] for r in recs})
    by = {e: [r for r in recs if r["engine"] == e] for e in engines}
    keys = ["detection_rate", "median_lead", "early_rate", "late_or_missed_rate", "precision", "recall"]
    vals = {k_: [] for k_ in keys}
    for _ in range(n_boot):
        s = rng.choice(engines, len(engines), replace=True)
        rr = [r for e in s for r in by[e]]
        sm = summarise(rr, trig)
        for k_ in keys:
            vals[k_].append(sm[k_])
    return {k_: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k_, v in vals.items()}


results = {"protocol": {"engines": "100 engines of train_FD001.txt (test files never read)", "folds": folds,
                        "fold_seed": FOLD_SEED, "inner_validation_engines_per_fold": 16, "seeds": SEEDS,
                        "persistence_m_policy": M_PERSIST, "k_spc": K_SPC, "mc_samples": MC_SAMPLES,
                        "triggers": {"simple": "latch at first cycle with predicted RUL <= R_acc",
                                     "confident": "latch at first cycle with predicted RUL + MC-dropout SD <= R_acc (outside the Boundary band)"},
                        "early": "true RUL at latch > fold-training q90 of acceleration RUL",
                        "late_or_missed": "true RUL at latch < fold-training q10, or never latched before failure",
                        "validation_only_models": str(MODELS.relative_to(ROOT))},
           "folds": fold_meta, "models": {}}
for mt in ("lstm", "datf"):
    recs = [r for r in engine_records if r["model"] == mt]
    trigs = [f"{t}_g{g}" for g in GATES for t in (("simple", "confident", "alert") if mt == "lstm" else ("simple",))]
    results["models"][mt] = {}
    for trig in trigs:
        per_fold = []
        for k in range(1, N_FOLDS + 1):
            fr = [r for r in recs if r["fold"] == k]
            per_seed = [summarise([r for r in fr if r["seed"] == s], trig) for s in SEEDS]
            per_fold.append({"fold": k, "R_acc": fold_meta[k - 1]["R_acc"],
                             **{key: float(np.mean([ps[key] for ps in per_seed])) for key in
                                ("detection_rate", "median_lead", "early_rate", "late_or_missed_rate", "precision", "recall",
                                 "engines_with_contradiction_after_latch", "latched_while_sensor_healthy_engines")},
                             "median_lead_seed_range": [min(ps["median_lead"] for ps in per_seed), max(ps["median_lead"] for ps in per_seed)]})
        results["models"][mt][trig] = {"pooled": summarise(recs, trig), "per_fold": per_fold, "bootstrap_ci95": boot_ci(recs, trig)}
    results["models"][mt]["raw_unlatched"] = {
        "precision": sum(r["raw_tp"] for r in recs) / max(sum(r["raw_tp"] + r["raw_fp"] for r in recs), 1),
        "recall": sum(r["raw_tp"] for r in recs) / max(sum(r["raw_tp"] + r["raw_fn"] for r in recs), 1),
        "returns_above_boundary_per_engine": dist([r["raw_returns_above_boundary"] for r in recs]),
        "boundary_share_of_cycles": sum(r["boundary_cycles"] for r in recs) / sum(r["cycles"] for r in recs),
        "abs_err_last_50_cycles": dist([r["abs_err_last_50_cycles"] for r in recs])}
# failure cases (LSTM, confident trigger): early latches / late-or-missed, aggregated over seeds
lr = [r for r in engine_records if r["model"] == "lstm"]
cases = {}
for r in lr:
    c = cases.setdefault(r["engine"], {"engine": r["engine"], "fold": r["fold"], "life": r["life"], "R_acc": r["R_acc"],
                                        "seeds_early": 0, "seeds_late_or_missed": 0, "leads": [], "contradicted": 0,
                                        "latched_sensor_healthy": 0})
    c["seeds_early"] += int(r[f"{PRIMARY}_early"]); c["seeds_late_or_missed"] += int(r[f"{PRIMARY}_late"])
    c["leads"].append(r[f"{PRIMARY}_lead"]); c["contradicted"] += int(r[f"{PRIMARY}_contradicted_cycles_after_latch"] > 0)
    c["latched_sensor_healthy"] += int(r[f"{PRIMARY}_latched_while_sensor_healthy_cycles"] > 0)
    c.setdefault("ungated_leads", []).append(r["confident_g0_lead"])
    c.setdefault("first_alert_leads", []).append(r["alert_g29_lead"])
failure_cases = sorted([c for c in cases.values() if c["seeds_early"] or c["seeds_late_or_missed"] or c["contradicted"]],
                       key=lambda c: -(c["seeds_early"] + c["seeds_late_or_missed"] + c["contradicted"]))
results["failure_cases_lstm_primary"] = {"trigger": PRIMARY, "cases": failure_cases}
R_all = [f["R_acc"] for f in fold_meta]
results["R_acc_across_folds"] = {"values": R_all, "min": min(R_all), "max": max(R_all), "mean": float(np.mean(R_all))}
(OUT / "near_failure_cv_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
with open(OUT / "near_failure_cv_engine_records.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(engine_records[0])); wtr.writeheader(); wtr.writerows(engine_records)
with open(OUT / "near_failure_cv_folds.csv", "w", newline="") as f:
    keys = [k_ for k_ in fold_meta[0] if k_ not in ("held_out", "inner_val")]
    wtr = csv.DictWriter(f, fieldnames=keys); wtr.writeheader(); wtr.writerows([{k_: m[k_] for k_ in keys} for m in fold_meta])

# ================= plots =================
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, mt, trig, title in ((axs[0], "lstm", PRIMARY, "LSTM, confident latch"), (axs[1], "datf", "simple_g29", "DATF-Net, simple latch")):
    data = [[r[f"{trig}_lead"] for r in engine_records if r["model"] == mt and r["fold"] == k and r[f"{trig}_lead"] is not None]
            for k in range(1, N_FOLDS + 1)]
    ax.boxplot(data, tick_labels=[str(k) for k in range(1, N_FOLDS + 1)], patch_artist=True,
               boxprops=dict(facecolor="#cfe0f6", color=C_BLUE), medianprops=dict(color=C_BLUE))
    for k in range(1, N_FOLDS + 1):
        ax.plot([k - 0.35, k + 0.35], [fold_meta[k - 1]["R_acc"]] * 2, color="black", lw=1)
    ax.set(title=f"{title} (cycle ≥ 29): true RUL at latch", xlabel="Fold (black line = fold R_acc)",
           ylabel="True RUL at latch (cycles)")
save_fig_name = "15_nfcv_lead_by_fold.png"
fig.tight_layout(); fig.savefig(OUT / save_fig_name, dpi=130); plt.close(fig)

ex = [c["engine"] for c in failure_cases[:4]] or list(state_examples)[:4]
fig, axs = plt.subplots(1, len(ex), figsize=(3.2 * len(ex), 3), sharey=True, squeeze=False)
for ax, u in zip(axs[0], ex):
    e = state_examples[u]; n = life[u]; true = n - 1 - e["cyc"]
    ax.fill_between(true, e["p"] - e["unc"], e["p"] + e["unc"], color=C_BLUE, alpha=0.18, label="Prediction ± MC SD")
    ax.plot(true, e["p"], color=C_BLUE, lw=1.2, label="Predicted RUL (seed 42)")
    ax.plot(true, np.minimum(true, 150), color=C_GREY, lw=1, ls="--", label="True RUL")
    ax.axhline(e["R"], color=S_CRIT, lw=1, ls=":", label=f"R_acc {e['R']:.0f}")
    if e["degr"] is not None:
        ax.axvline(n - 1 - e["degr"], color=S_WARN, lw=1, ls="--", label="Degrading declared")
    ax.set_xlim(min(n, 150), 0); ax.set_ylim(0, 150)
    ax.set(title=f"Engine {u} (fold {e['fold']})", xlabel="True RUL")
axs[0][0].set_ylabel("RUL (cycles)"); axs[0][0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "16_nfcv_failure_case_examples.png", dpi=130); plt.close(fig)

print(json.dumps({"folds": [{k_: v for k_, v in m.items() if k_ not in ("held_out", "inner_val")} for m in fold_meta]}, indent=1, default=float))
for mt in results["models"]:
    for trig, v in results["models"][mt].items():
        if trig == "raw_unlatched":
            print(mt, "raw", json.dumps(v, default=float)[:600]); continue
        print(mt, trig, "POOLED", json.dumps({k_: v["pooled"][k_] for k_ in v["pooled"] if k_ != "lead"}, default=float))
        print("   lead", json.dumps(v["pooled"]["lead"], default=float))
        print("   CI", json.dumps(v["bootstrap_ci95"], default=float))
        for pf in v["per_fold"]:
            print("   ", json.dumps(pf, default=float))
print("failure cases:", json.dumps(failure_cases[:12], default=float))
