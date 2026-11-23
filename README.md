# Industrial AI Predictive Maintenance – Remaining Useful Life (RUL)

Degradation-aware RUL prediction for turbofan engines using the **NASA C-MAPSS FD001** dataset.
Working pipeline: **LSTM baseline** → FastAPI backend (runs all inference) ⇄ HTTP ⇄ Streamlit dashboard.
**DATF-Net** (research model: TCN encoder + degradation-aware branch + gated fusion) is trained and evaluated in a separate
experiment pipeline (see "DATF-Net experiments"); it is **not** used by the API or dashboard.

## Architecture

```
 data/raw/*.txt ──► load_data ──► preprocess ──► train.py ──► models/checkpoints/lstm_best.pt
 (C-MAPSS FD001)   (RUL labels,   (engine split,   │           data/processed/preprocessor.joblib
                    clipping)      scaler, windows)│           outputs/training_history.json
                                                   ▼
                              predict.py (Predictor: same preprocessing as training)
                               ▲                    ▲
                    evaluate.py            backend/main.py  (FastAPI – the ONLY process that loads the model)
                (metrics.json, CSV)                 ▲
                                                    │  HTTP/JSON  (RUL_API_URL, default http://127.0.0.1:8000)
                                                    ▼
                                       dashboard/api_client.py ◄── dashboard/app.py (Streamlit + Plotly)
```

The dashboard contains no model, dataset or preprocessing code: it imports nothing from `src/` (enforced by
`tests/test_dashboard_client.py`). Every number it shows comes from the API:

| Dashboard feature | API call |
|---|---|
| backend/model status, health-state parameters, default sensor channels | `GET /health` |
| engine list / sensor history / true RUL | `GET /engines`, `GET /engines/{id}` |
| predicted RUL, uncertainty, data-driven health state + model/sensor evidence, explanation, sensor trends | `POST /predict` |
| per-cycle RUL trajectory chart | `POST /trajectory` |
| simulated demo engine | `GET /simulate` → readings sent to `POST /predict` and `POST /trajectory` |
| Model Performance tab | `GET /metrics`, `GET /metrics/history`, `GET /metrics/test-predictions` |

If the backend cannot be reached (refused/timed-out connection) the dashboard shows "Backend unavailable" with
the start command and renders no data; API errors (404/422/503) and malformed responses are shown as messages.
Responses are cached in the dashboard for up to 5 minutes (performance data: 1 minute); failed calls are not cached,
so the dashboard recovers as soon as the backend is back.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# GPU build of PyTorch (falls back to CPU automatically if CUDA is unavailable):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset

Download NASA C-MAPSS ("Turbofan Engine Degradation Simulation Data Set") from the NASA Prognostics Data
Repository and place these three files in `data/raw/`:

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

If they are missing, the loaders raise an explicit error; nothing is substituted. The dashboard's
**simulated demo** mode is separate and always labelled as simulated.

## Run

```powershell
python -m src.train          # trains LSTM (GPU if available), saves checkpoint + history
python -m src.evaluate       # real metrics on validation engines and the official test set
python -m pytest tests -q    # tests (includes an end-to-end test that starts a real backend process)
```

Start the system (two terminals, from the project folder; `.venv` need not be activated):

```powershell
# Terminal 1 – backend (loads the model; API docs at http://127.0.0.1:8000/docs)
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000

# Terminal 2 – dashboard (http://localhost:8501)
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Backend URL configuration: the dashboard reads `RUL_API_URL` (default `http://127.0.0.1:8000`) and
`RUL_API_TIMEOUT` in seconds (default 30). Example for a backend on another port or machine:

```powershell
$env:RUL_API_URL = "http://127.0.0.1:8010"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Options: `python -m src.train --seq-len 30 --rul-clip 125 --epochs 60 --patience 10 --hidden 64`.
Training is separate from inference: the dashboard/API only load the saved checkpoint.

## Dashboard (Engine Health Monitor)

Dark "control-room" UI (`dashboard/app.py`, components in `dashboard/components.py`, design tokens and chart styling
in `dashboard/theme.py`, Streamlit theme in `.streamlit/config.toml`). Sidebar navigation with backend status:

| Page | Content |
|---|---|
| Overview | RUL (dominant), health state, uncertainty, engine; RUL trajectory with current cycle and projected failure; degradation trend; sensor status |
| Engine Monitor | engine selector, data mode (real / simulation), Run prediction; current status; one sensor chart at a time (S2–S21), channel comparison, raw readings |
| RUL Analysis | prediction with uncertainty range (MC-dropout, not calibrated), ground truth or scenario comparison, large trajectory, health timeline, degradation analysis |
| Simulation | what-if scenarios (early-degradation / late-degradation / end-of-life scenario RUL bands, seed, cycles); reference engine vs simulated engine; always labelled SIMULATED |
| Model Performance | 5-seed research comparison read from `outputs/experiments/summary.json` (local file), paired bootstrap table, ablations; deployed-model metrics from the API |
| About | project, data, models, architecture, technology |

Live predictions always come from the LSTM baseline served by the API; DATF-Net results are shown as offline research
results only. Derived displays are computed from API data and state their rule (degradation trend = slope of the
predicted RUL over the last 30 cycles; sensor "stable" = |drift| < 0.5 training-σ).

## Health states (data-driven)

Implemented in `src/health_state.py`; frozen parameters in `data/processed/health_state_params.json`
(`python -m src.health_state_calibrate`, 80 training engines only). Evidence:
`outputs/health_threshold_analysis/analysis_report.md`, `cv_validation_report.md`,
`near_failure_cv/near_failure_cv_report.md`.

Health states are derived from FD001 training-data analysis. Degrading is identified using an engine-relative sensor
degradation index. Near Failure uses a fleet-level RUL acceleration boundary of 45 cycles. The 45-cycle boundary is
not an individual-engine failure threshold or safety limit. The 20-cycle persistence requirement is an implementation
policy chosen from validation analysis.

| State | Rule (per engine, evaluated over its whole history; 0-based history index) |
|---|---|
| Insufficient History | index 0–28 (index 29 is the first complete 30-cycle window); no latch possible |
| Healthy / Baseline-consistent | sensor index has not exceeded 0.641 for 20 consecutive cycles, prediction clearly above 45 |
| Degrading | sensor index (vs the engine's first 10 cycles, 10-cycle mean) > 0.641 for 20 consecutive cycles; latched |
| Boundary / Uncertain | \|predicted RUL − 45\| ≤ MC-dropout uncertainty |
| Near Failure | first index ≥ 29 with predicted RUL + uncertainty ≤ 45; latched, current prediction always shown |

Precedence of the single displayed state: Insufficient History > Near Failure (latched) > Boundary / Uncertain >
Degrading > Healthy. Model evidence and sensor evidence are returned and displayed separately; differences are
exposed in `disagreement` (`conflict` when the model is at/below the boundary but sensors are still at baseline,
`level_difference` when the model says Near Failure and the sensors say Degrading). After a latch,
`current_prediction_above_boundary` is true whenever the current prediction is above 45.

`src.predict.health_status` (old fixed bands) remains only because the frozen `src/evaluate.py` imports it to reproduce
the historical `health_status` column of `outputs/test_predictions.csv`; it is not used by the API, the dashboard or
the health-state logic, and that column is not served by the API.

## Model summary

* **Preprocessing**: constant channels dropped (data-driven, training engines only); StandardScaler fitted on
  training engines only; 80/20 split **by engine**; RUL = last cycle − cycle, clipped (default 125); sliding windows of
  30 cycles (stride 1). Test: one window at each engine's final recorded cycle (front-padded if shorter).
* **LSTM**: 2-layer LSTM (64 units) → FC 64 → 32 → 1, dropout 0.2, MSE on RUL/clip, Adam, ReduceLROnPlateau, early stopping on validation loss.
* **Uncertainty**: Monte-Carlo dropout spread — see "Uncertainty estimate" below. An *approximate, uncalibrated* indicator.
* **Health state**: data-driven, see "Health states" below. (The former fixed RUL bands were removed.)
* **Explanation**: predicted RUL + status rule + sensors drifting most from the engine's early baseline (correlation, not causal attribution).

## Uncertainty estimate (MC dropout)

The point prediction is always the deterministic model output (dropout off). The uncertainty is the standard
deviation (ddof = 1) of 30 extra predictions in which the dropout layers of the fully-connected head
(p = 0.2, as in training) are active; the LSTM is run once in eval mode, as before.

* **Reproducible:** the 30 dropout masks are drawn from a local `torch.Generator` seeded with `MC_DROPOUT_SEED`
  (`src/config.py`), so identical inputs give identical uncertainty. Previously the masks came from the unseeded
  global RNG, so every request gave a slightly different value.
* **No side effects:** masks are applied explicitly; the global RNG is untouched and the shared model is never
  switched to train mode (the old code toggled dropout on the shared model, which is unsafe under concurrent API
  requests).
* **Not forced:** it is still a genuine spread over random masks (values 2.7–13.1 cycles across the 100 test
  engines). With 2,000–5,000 masks it agrees with the old method within sampling error.
* **Limitations:** the fixed seed freezes the sampling error of a 30-sample estimate (about ±13% of the value)
  instead of re-drawing it; every engine uses the same 30 masks; MC dropout on the head only captures part of
  the model uncertainty; the values are not calibrated (not a confidence interval). The `uncertainty` column in
  `outputs/test_predictions.csv` was written by the old unseeded method and was not regenerated.

## Simulated demo mode

`GET /simulate?stage=normal|warning|critical&seed=…&n_cycles=…` produces **simulated** readings that are always
labelled as such and never used in metrics:

1. From the real FD001 training file, the mean reading of every sensor channel is computed as a function of true
   RUL (smoothed; RUL values seen in fewer than 10 engines use the healthy plateau), plus two noise levels per
   channel: engine-to-engine offset and cycle-to-cycle noise.
2. The chosen stage fixes a **scenario RUL** at the last observed cycle, drawn reproducibly from the seed within a
   fixed scenario band: early degradation 80–110, late degradation 33–42, end of life 5–16 cycles (API stage keys
   `normal` / `warning` / `critical`; scenario definitions only, not health states).
3. Cycle *c* of *n* has scenario RUL `rul_end + (n − c)`; its readings are profile(RUL) + one engine offset +
   cycle noise. Channels that are constant in FD001 keep their real constant value.

The sensors and the scenario RUL are therefore consistent by construction; the model's prediction is **not**
forced to match. Over 30 seeds (180 cycles) the model's predicted RUL fell within the scenario band ±15 cycles for
83 % (early), 90 % (late) and 100 % (end-of-life) of simulated engines. Simulated engines get their health state from
the same data-driven rules as real engines.
The dashboard shows scenario RUL, predicted RUL and their difference, and says so when the statuses differ.

## Results (actual run: RTX GPU, seed 42, single split, best epoch 11 of 21)

| Set | MAE (cycles) | RMSE (cycles) |
|---|---|---|
| Validation (20 held-out training engines, RUL clipped at 125) | 10.37 | 14.40 |
| Official FD001 test (100 engines, vs unclipped RUL_FD001.txt) | 11.82 | 16.07 (NASA score 504.6) |

Single run, no confidence intervals; regenerate with `python -m src.train` and `python -m src.evaluate`.

Note: the scaler is a small NumPy re-implementation of `StandardScaler` (same maths) because scipy/scikit-learn DLLs were
blocked by a Windows Application Control policy on the development machine; scikit-learn is therefore not required.

**No pandas.** The same policy blocks pandas' compiled parser DLL (`pandas._libs.parsers`), so the project does not use
pandas at all. Data is held as a `Table` (a dict of NumPy column arrays, `src/data/table.py`); files are read with
`numpy.loadtxt`, CSVs with the `csv` module, and dashboard tables are rendered as escaped HTML
(`dashboard/table_view.py`), because `st.dataframe`/`st.table` convert data with pandas in a live Streamlit session.
The NumPy pipeline was checked to reproduce the pandas-era windows, predictions and metrics exactly, and
`tests/test_no_pandas.py` fails if any core module starts importing pandas again.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | service, model and dataset status |
| `GET /engines`, `GET /engines/{id}?last_n=50` | test-set engines / one engine's last `last_n` readings and true RUL |
| `POST /predict` | body `{"engine_id": 1}` **or** `{"readings": [ {cycle?, op_setting_1…sensor_21}, … ]}` → RUL at last cycle, status, uncertainty, explanation |
| `POST /trajectory` | same body as `/predict` → predicted RUL at every supplied cycle (+ true RUL for test engines) |
| `GET /simulate?stage=normal&seed=0&n_cycles=180` | **simulated** readings for demo mode at a chosen stage, with the scenario RUL per cycle (flagged `"simulated": true`; `n_cycles` 10–1000) |
| `GET /metrics` | contents of `outputs/metrics.json` |
| `GET /metrics/history`, `GET /metrics/test-predictions` | training loss curves / per-engine test predictions |

Returns 503 with instructions if the model or dataset is missing, 404 for unknown engines or missing result files,
422 for invalid input (including readings with missing fields, partial or duplicate `cycle` values).

## Demo procedure (≈5 min)

1. Start the backend, then the dashboard (see "Run"). **Overview**: engine count, selected engine, predicted RUL, health, model status.
2. Pick an engine with a low true RUL (e.g. from `outputs/test_predictions.csv`) → **RUL Prediction**: big RUL number, status, explanation, actual (RUL_FD001) vs predicted, trajectory.
3. **Sensor Analytics**: choose channels, compare normalised trends.
4. **Model Performance**: MAE/RMSE, loss curves, predicted-vs-actual scatter.
5. Sidebar → *Demo: simulated engine* → choose stage, seed and cycles → *Load sample engine trajectory*. Nothing
   changes until the button is clicked. The simulated engine's data-driven health state is shown next to its
   scenario RUL.
6. Open `/docs` of the API and run `POST /predict` with `{"engine_id": 34}`.

## DATF-Net experiments

Files: `src/models/datf_net.py` (model), `src/data/degradation.py` (engine baselines, HI targets), `src/train_datf.py`
(training loop), `src/experiments.py` (protocol), `tests/test_datf.py`. Results: `outputs/experiments/` (one folder
per run and seed with `run.json`, `epochs.log`, test predictions; `tune/selection.json`; `summary.md`/`summary.json`).
Checkpoints: `models/checkpoints/datf/`. The baseline files are never written (verified by SHA-256 before/after).

```powershell
.\.venv\Scripts\python.exe -m src.experiments tune     # lambda grid, 5 seeds, validation only
.\.venv\Scripts\python.exe -m src.experiments final    # LSTM + DATF-Net + 7 ablations, 5 seeds, test evaluated once
.\.venv\Scripts\python.exe -m src.experiments control  # information-matched LSTM control, 5 seeds
.\.venv\Scripts\python.exe -m src.experiments report   # summary tables + paired bootstrap
```

**Model** (54,626 parameters; LSTM 60,801): input window `(B,30,17)` + engine baseline `b (B,17)` = mean of the engine's
first 10 observed cycles. Causal dilated TCN (3 blocks, d=48, dilations 1/2/4, receptive field 29) → `H (B,30,48)`.
Degradation branch: `D_t = ReLU(W(x_t − b))`, health index `HI_t = σ(w·D_t + c)`, least-squares window slope →
`z_d (B,48)`. Attention pooling over `H` with the query from `z_d`, gated fusion, head 48→32→1.
**Health index** = normalised RUL: trained towards `min(RUL_t,125)/125` for every cycle of the window (1 = healthy,
0 = failure, non-increasing). Loss = MSE(RUL/125) + λ_hi·MSE(HI) + λ_mono·mean ReLU(HI_{t+1} − HI_t).

**Protocol:** same 80/20 engine split, scaler, windows and RUL cap as the baseline (read from the saved preprocessor);
identical optimiser, schedule, early stopping and seeding order as `src/train.py` — a fresh LSTM run with seed 42
reproduces the protected baseline exactly (val loss identical at all 21 epochs; test MAE/RMSE/NASA 11.82/16.07/504.6).
λ_hi ∈ {0.1, 0.3, 1}, λ_mono ∈ {0, 0.1} selected by mean validation RMSE over 5 seeds (42–46) with all test loaders
disabled (enforced by a test) → λ_hi = 0.1, λ_mono = 0.1 (λ_mono = 0 was within 0.01 RMSE). Test evaluated once per
final run. Early stopping uses validation engines only.

**Results (5 seeds, mean ± std; test = 100 engines vs unclipped RUL_FD001):**

| Run | Val RMSE | Test MAE | Test RMSE | Test NASA |
|---|---|---|---|---|
| LSTM (fresh, same protocol) | 14.39 ± 0.39 | 11.65 ± 0.23 | 16.02 ± 0.35 | 427 ± 72 |
| LSTM + engine baseline, input `[x, x − b]` (information-matched control) | 11.59 ± 0.55 | 12.02 ± 0.72 | 16.34 ± 0.92 | 377 ± 41 |
| **DATF-Net (full)** | 11.43 ± 0.27 | 10.66 ± 0.41 | 14.89 ± 0.59 | 286 ± 23 |
| − degradation branch | 16.05 ± 0.44 | 12.30 ± 0.51 | 16.62 ± 0.74 | 423 ± 67 |
| − engine baseline (x instead of x − b) | 13.40 ± 0.23 | 11.28 ± 0.23 | 15.00 ± 0.45 | 312 ± 28 |
| − window slope | 11.86 ± 0.44 | 11.55 ± 0.38 | 15.50 ± 0.54 | 318 ± 22 |
| − HI supervision (λ = 0) | 11.30 ± 0.29 | 10.74 ± 0.50 | 14.95 ± 0.54 | 290 ± 25 |
| attention → last-step pooling | 11.18 ± 0.17 | 10.55 ± 0.32 | 14.41 ± 0.61 | 268 ± 22 |
| gated → concat fusion | 11.44 ± 0.29 | 10.79 ± 0.71 | 14.92 ± 0.85 | 289 ± 38 |
| TCN → LSTM encoder | 11.50 ± 0.44 | 10.66 ± 0.43 | 14.51 ± 0.51 | 281 ± 19 |

Paired bootstrap over the 100 test engines (ΔRMSE, 95% CI; negative = first is better):

| Comparison | Information seen by both | ΔRMSE | 95% CI |
|---|---|---|---|
| DATF-Net − LSTM | – (DATF also sees b) | −1.13 | [−3.25, +0.90] |
| LSTM + b − LSTM | – (control also sees b) | +0.32 | [−1.80, +2.30] |
| **DATF-Net − LSTM + b** | window + baseline b | −1.45 | [−3.02, +0.01] |
| **DATF-Net without b − LSTM** | window only | −1.01 | [−2.55, +0.51] |

Ablations vs full DATF-Net: removing the degradation branch +1.73 [−0.20, +3.70]; removing the slope +0.61
[−0.05, +1.24]; all other ablations' CIs are centred near 0 (see `outputs/experiments/summary.md`).

**Information-matched control.** Same architecture and hyper-parameters as the baseline LSTM (65,153 parameters;
only the first layer's input grows from 17 to 34), same split, scaler, windows, targets, optimiser, schedule and
early stopping; input `[x_t, x_t − b]` with the same `b` DATF-Net uses. It was pre-specified and not tuned, so no
model selection (and no test use) was involved. Code: `src/models/lstm_control.py`; runs:
`outputs/experiments/final/lstm_info_matched/`.

**Interpretation (evidence only):**
* *Validation:* the engine-baseline information explains the gain. The control reaches 11.59 ± 0.55 (DATF-Net 11.43
  ± 0.27, LSTM 14.39), and DATF-Net without `b` falls back to 13.40.
* *Test:* the information does not help. The control is no better than the LSTM (16.34 vs 16.02), and DATF-Net
  without `b` scores the same as full DATF-Net (15.00 vs 14.89). The test advantage of DATF-Net (about 1–1.5 RMSE)
  appears with or without `b`, so it is associated with the architecture (degradation branch / slope pathway), not
  the extra information — but neither information-matched comparison excludes zero at 95% (upper bounds +0.01 and
  +0.51).
* *Validation and test disagree.* The validation set (all windows of 20 complete engines) rewards the baseline
  information; the test set (the last window of 100 truncated engines) does not. With these data DATF-Net is **not
  shown to be superior**; the most defensible statement is "a consistent but statistically inconclusive test
  improvement of about 1 RMSE that does not depend on the extra baseline information".
* The control stopped very early (best epochs 2–6 vs 6–11 for the LSTM) and varied more across seeds on test
  (15.68–17.90), consistent with quickly fitting engine-specific baseline information. It used the baseline's
  untuned hyper-parameters by design; DATF-Net's λ values were tuned on validation (λ = 0 scored the same on test).
* Health index: tracks normalised RUL (validation correlation 0.90; mean 0.88 when healthy, 0.09 within 10 cycles
  of failure) but is noisy cycle-to-cycle (≈48% of steps increase); λ_mono = 0.1 did not enforce monotonicity.

## Status

**Completed**: data loading/preprocessing, LSTM training/eval, saved checkpoint & scaler, inference (with MC-dropout uncertainty), FastAPI, Streamlit dashboard, simulated demo mode, tests.
**Research / pending**: DATF-Net is trained and evaluated (above) but not integrated into the API or dashboard. No
claim of novelty or superiority beyond the reported evidence. The information-matched LSTM control is done (see above).
Open: simplified variant, monotone HI, engine-level cross-validation, FD002–FD004.

## Known limitations & future work

* FD001 only (single operating condition); results may not transfer to FD002–FD004.
* The 45-cycle Near-Failure boundary is fleet-level (not an individual-engine threshold or safety limit); the
  20-cycle persistence is a policy choice; uncertainty is uncalibrated.
* Test RUL ground truth exists only at each engine's last cycle; the per-cycle "actual" line in the dashboard is derived from it.
* Future: FD002–FD004, information-matched controls for DATF-Net, calibrated uncertainty, drift monitoring.
