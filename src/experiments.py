"""DATF-Net experiment protocol: tuning (validation only) -> final multi-seed runs + ablations -> report.

  python -m src.experiments tune     # lambda_hi x lambda_mono grid for full DATF-Net, 5 seeds, VALIDATION ONLY
  python -m src.experiments final    # LSTM + DATF-Net + ablations, 5 seeds, selected lambdas, test evaluated once
  python -m src.experiments control  # information-matched LSTM control ([x, x - b] input), 5 seeds, same protocol
  python -m src.experiments report   # summary tables + paired bootstrap on test engines

Outputs: outputs/experiments/<stage>/<run>/seed_<s>/run.json (+ test_predictions.csv in the final stage),
         outputs/experiments/tune/selection.json, outputs/experiments/summary.{json,md}
Checkpoints: models/checkpoints/datf/<stage>/<run>_seed<s>.pt
The baseline files (models/checkpoints/lstm_best.pt, data/processed/preprocessor.joblib, outputs/metrics.json,
outputs/test_predictions.csv, outputs/training_history.json) are only read, never written.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from src import config as C
from src.data.degradation import BASELINE_CYCLES, load_test_split, load_train_val
from src.data.preprocess import Preprocessor
from src.evaluate import mae, nasa_score, rmse
from src.train_datf import RunSpec, TrainSettings, load_checkpoint, predict, save_json, train_run

SEEDS = [42, 43, 44, 45, 46]
EXP_DIR = C.OUTPUT_DIR / "experiments"
CKPT_DIR = C.CHECKPOINT_DIR / "datf"
TUNE_GRID = [(lh, lm) for lh in (0.1, 0.3, 1.0) for lm in (0.0, 0.1)]


def control_specs():
    """Information-matched control: baseline LSTM hyper-parameters (no tuning), input [x, x - b]."""
    return [RunSpec("lstm_info_matched", "lstm_info")]


def tune_specs():
    return [RunSpec(f"datf_lh{lh}_lm{lm}", "datf", {}, lh, lm) for lh, lm in TUNE_GRID]


def final_specs(lh: float, lm: float):
    """Ablations remove ONE component of the full model; lambdas are the ones selected on validation."""
    return [
        RunSpec("lstm_baseline", "lstm"),
        RunSpec("datf_full", "datf", {}, lh, lm),
        RunSpec("abl_no_degradation", "datf", {"use_degradation": False}, 0.0, 0.0),
        RunSpec("abl_no_baseline", "datf", {"use_baseline": False}, lh, lm),
        RunSpec("abl_no_slope", "datf", {"use_slope": False}, lh, lm),
        RunSpec("abl_no_hi_supervision", "datf", {}, 0.0, 0.0),
        RunSpec("abl_last_step_pooling", "datf", {"pooling": "last"}, lh, lm),
        RunSpec("abl_concat_fusion", "datf", {"fusion": "concat"}, lh, lm),
        RunSpec("abl_lstm_encoder", "datf", {"encoder": "lstm"}, lh, lm),
    ]


# ---------------- metrics ----------------
def val_metrics(model, val, clip, device) -> dict:
    pred, hi = predict(model, val["X"], val["b"], clip, device)
    m = {"n_windows": int(len(pred)), "n_engines": int(len(np.unique(val["unit"]))),
         "mae": mae(val["y"], pred), "rmse": rmse(val["y"], pred), "target": f"RUL clipped at {int(clip)}"}
    if hi is not None:
        tgt = val["hi_target"]
        m["health_index"] = {
            "definition": f"HI_t trained towards min(RUL_t, {int(clip)})/{int(clip)}; 1 = healthy, 0 = failure",
            "mae_vs_target": float(np.abs(hi - tgt).mean()),
            "corr_with_target_all_cycles": float(np.corrcoef(hi.ravel(), tgt.ravel())[0, 1]),
            "corr_last_cycle_hi_with_rul": float(np.corrcoef(hi[:, -1], val["y"])[0, 1]),
            "mean_hi_where_target_1": float(hi[tgt >= 0.999].mean()) if (tgt >= 0.999).any() else None,
            "mean_hi_where_rul_le_10": float(hi[tgt * clip <= 10].mean()) if (tgt * clip <= 10).any() else None,
            "frac_windows_non_increasing_tol_0.01": float(((np.diff(hi, axis=1)) <= 0.01).all(axis=1).mean()),
        }
    return m


def test_metrics(model, test, clip, device) -> tuple[dict, np.ndarray]:
    pred, _ = predict(model, test["X"], test["b"], clip, device)
    true = test["true_rul"]
    tc = np.minimum(true, clip)
    return ({"n_engines": int(len(true)), "mae": mae(true, pred), "rmse": rmse(true, pred),
             "nasa_score": nasa_score(true, pred), "mae_vs_clipped_true": mae(tc, pred),
             "rmse_vs_clipped_true": rmse(tc, pred),
             "target": "official RUL_FD001.txt (unclipped) at last recorded cycle"}, pred)


# ---------------- stages ----------------
def _run_dir(root, stage, name, seed):
    return root / stage / name / f"seed_{seed}"


def run_stage(stage: str, specs, seeds, settings_base: TrainSettings, exp_dir=EXP_DIR, ckpt_dir=CKPT_DIR,
              evaluate_test: bool = False, reuse: dict | None = None, log=print):
    prep = Preprocessor.load()  # read-only: same scaler / features / validation engines as the baseline
    clip = float(prep.rul_clip)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_train_val(prep, BASELINE_CYCLES)
    test = load_test_split(prep, BASELINE_CYCLES) if evaluate_test else None
    log(f"[{stage}] device {device}; train windows {train['X'].shape}, val windows {val['X'].shape}; "
        f"val engines {sorted(prep.val_units)}; test loaded: {test is not None}")
    for spec in specs:
        for seed in seeds:
            run_dir = _run_dir(exp_dir, stage, spec.name, seed)
            ckpt = ckpt_dir / stage / f"{spec.name}_seed{seed}.pt"
            if (run_dir / "run.json").exists():
                log(f"[{stage}] {spec.name} seed {seed}: already done, skipping")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            settings = TrainSettings(**{**settings_base.to_dict(),
                                        "lambda_hi": spec.lambda_hi or 0.0, "lambda_mono": spec.lambda_mono or 0.0})
            source = (reuse or {}).get((spec.name, seed))
            if source is not None:  # identical spec + seed already trained in the tuning stage
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source["ckpt"], ckpt)
                history = source["history"]
                note = f"checkpoint reused from tuning run {source['ckpt'].name} (same config and seed)"
            else:
                epoch_log = run_dir / "epochs.log"
                epoch_log.parent.mkdir(parents=True, exist_ok=True)
                with open(epoch_log, "w") as fh:
                    history = train_run(spec, train, val, seed, settings, clip, ckpt, device,
                                        extra_meta={"feature_cols": prep.feature_cols, "seq_len": prep.seq_len,
                                                    "baseline_cycles": BASELINE_CYCLES, "stage": stage},
                                        log=lambda s: fh.write(s + "\n"))
                note = None
            model, ck = load_checkpoint(ckpt, len(prep.feature_cols), prep.seq_len, device)
            record = {"stage": stage, "run": spec.name, "seed": seed, "model_type": spec.model_type,
                      "datf_overrides": spec.datf, "datf_config": ck["datf_config"], "settings": settings.to_dict(),
                      "split": {"train_engines": int(len(np.unique(train["unit"]))),
                                "val_engines": sorted(int(u) for u in prep.val_units),
                                "source": "data/processed/preprocessor.joblib (baseline split, read-only)"},
                      "baseline_cycles": BASELINE_CYCLES, "best_epoch": ck["epoch"], "history": history,
                      "validation": val_metrics(model, val, clip, device), "note": note}
            if test is not None:
                record["test"], pred = test_metrics(model, test, clip, device)
                with open(run_dir / "test_predictions.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["engine_id", "true_rul", "predicted_rul"])
                    w.writerows(zip(test["engine_id"].tolist(), test["true_rul"].tolist(), pred.tolist()))
            save_json(run_dir / "run.json", record)
            v = record["validation"]
            msg = f"[{stage}] {spec.name} seed {seed}: best epoch {ck['epoch']}, val RMSE {v['rmse']:.2f} MAE {v['mae']:.2f}"
            if test is not None:
                msg += f" | test RMSE {record['test']['rmse']:.2f} MAE {record['test']['mae']:.2f}"
            log(msg + (f" ({history['train_seconds']}s)" if note is None else " (reused)"))


def load_runs(stage, exp_dir=EXP_DIR) -> dict:
    runs = {}
    for f in sorted((exp_dir / stage).glob("*/seed_*/run.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["run"], []).append(r)
    return runs


def select_lambdas(exp_dir=EXP_DIR, seeds=SEEDS) -> dict:
    """Pick (lambda_hi, lambda_mono) by mean VALIDATION RMSE over seeds. Test data is not read."""
    runs = load_runs("tune", exp_dir)
    table = []
    for spec in tune_specs():
        rs = [r for r in runs.get(spec.name, []) if r["seed"] in seeds]
        if len(rs) != len(seeds):
            raise RuntimeError(f"tuning incomplete for {spec.name}: {len(rs)}/{len(seeds)} seeds")
        assert all("test" not in r for r in rs), "tuning runs must not contain test metrics"
        v = np.array([r["validation"]["rmse"] for r in rs])
        table.append({"run": spec.name, "lambda_hi": spec.lambda_hi, "lambda_mono": spec.lambda_mono,
                      "val_rmse_mean": float(v.mean()), "val_rmse_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "val_mae_mean": float(np.mean([r["validation"]["mae"] for r in rs]))})
    best = min(table, key=lambda t: (round(t["val_rmse_mean"], 6), t["lambda_hi"], t["lambda_mono"]))
    sel = {"criterion": "lowest mean validation RMSE over seeds (ties -> smaller lambdas)", "seeds": seeds,
           "selected": {"lambda_hi": best["lambda_hi"], "lambda_mono": best["lambda_mono"], "run": best["run"]},
           "grid": table}
    save_json(exp_dir / "tune" / "selection.json", sel)
    return sel


def bootstrap_rmse_diff(pa: np.ndarray, pb: np.ndarray, true: np.ndarray, n: int = 10_000, seed: int = 0):
    """Paired bootstrap over test engines. pa, pb: (seeds, engines). Statistic: mean-over-seeds RMSE(a) - RMSE(b)."""
    rng = np.random.default_rng(seed)
    def stat(idx):
        ra = np.sqrt(((pa[:, idx] - true[idx]) ** 2).mean(axis=1)).mean()
        rb = np.sqrt(((pb[:, idx] - true[idx]) ** 2).mean(axis=1)).mean()
        return ra - rb
    obs = stat(np.arange(len(true)))
    boots = np.array([stat(rng.integers(0, len(true), len(true))) for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rmse_diff": float(obs), "ci95": [float(lo), float(hi)], "frac_boot_below_0": float((boots < 0).mean())}


def report(exp_dir=EXP_DIR, seeds=SEEDS) -> dict:
    runs = load_runs("final", exp_dir)
    sel = json.loads((exp_dir / "tune" / "selection.json").read_text())
    rows, preds, true = {}, {}, None
    for name, rs in runs.items():
        rs = sorted((r for r in rs if r["seed"] in seeds), key=lambda r: r["seed"])
        if len(rs) != len(seeds):
            continue
        agg = lambda part, key: (float(np.mean([r[part][key] for r in rs])), float(np.std([r[part][key] for r in rs], ddof=1)))  # noqa: E731
        rows[name] = {"seeds": [r["seed"] for r in rs], "n_parameters": rs[0]["history"]["n_parameters"],
                      "best_epochs": [r["best_epoch"] for r in rs],
                      "val_mae": agg("validation", "mae"), "val_rmse": agg("validation", "rmse"),
                      "test_mae": agg("test", "mae"), "test_rmse": agg("test", "rmse"),
                      "test_nasa": agg("test", "nasa_score"), "test_rmse_vs_clipped": agg("test", "rmse_vs_clipped_true"),
                      "per_seed_test_rmse": [r["test"]["rmse"] for r in rs]}
        if "health_index" in rs[0]["validation"]:
            rows[name]["hi_corr_with_target"] = float(np.mean(
                [r["validation"]["health_index"]["corr_with_target_all_cycles"] for r in rs]))
        P = []
        for r in rs:
            with open(exp_dir / "final" / name / f"seed_{r['seed']}" / "test_predictions.csv", newline="") as f:
                d = list(csv.DictReader(f))
            P.append([float(x["predicted_rul"]) for x in d])
            true = np.array([float(x["true_rul"]) for x in d])
        preds[name] = np.array(P)
    comparisons = {}
    # information-matched pairs: (DATF-Net, LSTM + [x, x-b]) both see the baseline; (DATF-Net without baseline,
    # LSTM) both see only the 30-cycle window
    for a, b in (("datf_full", "lstm_baseline"), ("lstm_info_matched", "lstm_baseline"),
                 ("datf_full", "lstm_info_matched"), ("abl_no_baseline", "lstm_baseline")):
        if a in preds and b in preds:
            comparisons[f"{a}_minus_{b}"] = bootstrap_rmse_diff(preds[a], preds[b], true)
    for name in preds:
        if name.startswith("abl_") and "datf_full" in preds:
            comparisons[f"{name}_minus_datf_full"] = bootstrap_rmse_diff(preds[name], preds["datf_full"], true)
    summary = {"protocol": {"seeds": seeds, "selection": sel["selected"], "baseline_cycles": BASELINE_CYCLES,
                            "test_used_for": "final evaluation only (after lambda selection on validation)"},
               "protected_baseline_metrics": json.loads(C.METRICS_PATH.read_text()),
               "variants": rows, "paired_bootstrap_test_rmse": comparisons}
    save_json(exp_dir / "summary.json", summary)
    (exp_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(s) -> str:
    f = lambda t: f"{t[0]:.2f} ± {t[1]:.2f}"  # noqa: E731
    lines = ["# DATF-Net experiments (FD001)", "",
             f"Seeds {s['protocol']['seeds']}; lambdas selected on validation: {s['protocol']['selection']}; "
             f"test used only for final evaluation.", "",
             "| Run | Params | Val MAE | Val RMSE | Test MAE | Test RMSE | Test NASA | Test RMSE (clipped truth) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in s["variants"].items():
        lines.append(f"| {name} | {r['n_parameters']:,} | {f(r['val_mae'])} | {f(r['val_rmse'])} | {f(r['test_mae'])} | "
                     f"{f(r['test_rmse'])} | {r['test_nasa'][0]:.0f} ± {r['test_nasa'][1]:.0f} | {f(r['test_rmse_vs_clipped'])} |")
    b = s["protected_baseline_metrics"]["test"]
    lines += ["", f"Protected single-seed baseline (outputs/metrics.json, unchanged): test MAE {b['mae']:.2f}, "
              f"RMSE {b['rmse']:.2f}, NASA {b['nasa_score']:.1f}.", "",
              "## Paired bootstrap on the 100 test engines (RMSE difference, mean over seeds; negative = first is better)", "",
              "| Comparison | ΔRMSE | 95% CI |", "|---|---|---|"]
    for k, v in s["paired_bootstrap_test_rmse"].items():
        lines.append(f"| {k} | {v['rmse_diff']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "final", "control", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)  # noqa: E731
    settings = TrainSettings(epochs=args.epochs)
    if args.stage == "tune":
        run_stage("tune", tune_specs(), args.seeds, settings, evaluate_test=False, log=log)
        sel = select_lambdas(seeds=args.seeds)
        log(f"[tune] selected {sel['selected']}")
        for t in sel["grid"]:
            log(f"   {t['run']}: val RMSE {t['val_rmse_mean']:.3f} ± {t['val_rmse_std']:.3f}")
    elif args.stage == "final":
        sel = json.loads((EXP_DIR / "tune" / "selection.json").read_text())["selected"]
        reuse = {}
        for seed in args.seeds:
            d = _run_dir(EXP_DIR, "tune", sel["run"], seed)
            if (d / "run.json").exists():
                reuse[("datf_full", seed)] = {"ckpt": CKPT_DIR / "tune" / f"{sel['run']}_seed{seed}.pt",
                                              "history": json.loads((d / "run.json").read_text())["history"]}
        run_stage("final", final_specs(sel["lambda_hi"], sel["lambda_mono"]), args.seeds, settings,
                  evaluate_test=True, reuse=reuse, log=log)
    elif args.stage == "control":
        run_stage("final", control_specs(), args.seeds, settings, evaluate_test=True, log=log)
    else:
        s = report(seeds=args.seeds)
        sys.stdout.write(_markdown(s))


if __name__ == "__main__":
    main()
