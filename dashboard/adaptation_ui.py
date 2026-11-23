"""UI views for Mode 2 (External CSV Upload) and Mode 3 (Industry Adaptation).

Pandas-free: all tables use `table_view.html_table` or native markdown.
Reaches model and data strictly through `api_client.ApiClient` via HTTP.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from api_client import ApiClient, ApiError
from components import (
    banner_html,
    metric_card_html,
    note_html,
    render_metric_card,
    render_page_header,
    render_section_header,
    status_badge_html,
)
from table_view import html_table, records_to_columns
from theme import CSS, INFO, NEUTRAL_LINE, OFFLINE, ONLINE, SERIES, STATUS, style_fig


def _ensure_client(api: ApiClient) -> ApiClient:
    if not hasattr(api, "upload_dataset"):
        return ApiClient(getattr(api, "base_url", None))
    return api


def _get_active_dataset(api: ApiClient) -> dict[str, Any] | None:
    return st.session_state.get("active_dataset")


def render_mode2_upload(api: ApiClient):
    api = _ensure_client(api)
    render_page_header("EXTERNAL DATASET UPLOAD", "Upload, validate, and inspect external machine sensor CSV, TXT, or ZIP archives")

    st.markdown(
        "Upload a sensor dataset from your equipment (supports **.csv**, **.txt**, and **.zip** archives, including NASA C-MAPSS FD001–FD004 bundles). "
        "The system enforces strict ZIP Slip and size protections, identifies subsets, validates time-series continuity, and evaluates model safety constraints."
    )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        uploaded_file = st.file_uploader("Upload external CSV, TXT, or ZIP (Max 50 MB)", type=["csv", "txt", "zip"], key="mode2_file_upload")
    with c2:
        st.write("**Industrial Samples:**")
        if st.button("Load Industrial Pump", key="load_sample_pump"):
            sample_path = Path("data/sample_external/synthetic_pump_run_to_failure.csv")
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    data_bytes = f.read()
                with st.spinner("Uploading and analyzing sample pump dataset..."):
                    res = api.upload_dataset(data_bytes, sample_path.name)
                    st.session_state["active_dataset"] = res
                    st.success("Loaded synthetic industrial pump dataset.")
                    st.rerun()
    with c3:
        st.write("**Incompatible:**")
        if st.button("Load Incompatible Motor", key="load_sample_motor"):
            sample_path = Path("data/sample_external/synthetic_motor_incompatible.csv")
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    data_bytes = f.read()
                with st.spinner("Uploading and analyzing incompatible motor dataset..."):
                    res = api.upload_dataset(data_bytes, sample_path.name)
                    st.session_state["active_dataset"] = res
                    st.warning("Loaded incompatible motor dataset.")
                    st.rerun()
    with c4:
        st.write("**NASA Archive:**")
        if st.button("Load C-MAPSS ZIP", key="load_sample_cmapss_zip"):
            sample_path = Path("data/sample_external/sample_cmapss_fd001_fd002.zip")
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    data_bytes = f.read()
                with st.spinner("Decompressing, parsing, and verifying NASA C-MAPSS archive..."):
                    res = api.upload_dataset(data_bytes, sample_path.name)
                    st.session_state["active_dataset"] = res
                    st.success("Loaded NASA C-MAPSS multi-subset ZIP archive.")
                    st.rerun()

    # Handle file upload
    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            file_bytes = uploaded_file.getvalue()
            with st.spinner("Decompressing, analyzing, and validating dataset..."):
                try:
                    res = api.upload_dataset(file_bytes, uploaded_file.name)
                    st.session_state["active_dataset"] = res
                    st.session_state["last_uploaded_filename"] = uploaded_file.name
                    st.success(f"Successfully processed {uploaded_file.name}")
                except ApiError as e:
                    st.error(f"Upload failed: {str(e)}")

    active = _get_active_dataset(api)
    if not active:
        st.info("No external dataset uploaded yet. Upload a CSV/TXT/ZIP or click one of the sample buttons above.")
        return

    # NASA C-MAPSS Archive Multi-Subset Selector
    zip_meta = active.get("zip_metadata", {})
    if zip_meta.get("is_zip") and zip_meta.get("archive_type") == "nasa_cmapss":
        subsets = zip_meta.get("subsets_detected", [])
        active_sub = zip_meta.get("active_subset")
        sub_info_all = zip_meta.get("subsets_info", {})

        st.write("")
        render_section_header("NASA C-MAPSS Archive Detection", f"Archive contains {len(subsets)} sub-datasets")
        sub_col1, sub_col2 = st.columns([1, 2])
        with sub_col1:
            curr_idx = subsets.index(active_sub) if active_sub in subsets else 0
            sel_subset = st.radio("Select Active NASA Subset to Inspect / Predict:", subsets, index=curr_idx, horizontal=True)

            if sel_subset != active_sub:
                target_sub = sub_info_all.get(sel_subset, {})
                if target_sub.get("converted_csv_path"):
                    try:
                        updated = api.select_dataset_subset(sel_subset, target_sub["converted_csv_path"], target_sub.get("rul_file"))
                        active["saved_path"] = updated["saved_path"]
                        active["validation_report"] = updated["validation_report"]
                        zip_meta["active_subset"] = sel_subset
                        st.session_state["active_dataset"] = active
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to switch subset: {str(e)}")

        with sub_col2:
            curr_info = sub_info_all.get(zip_meta.get("active_subset", ""), {})
            compat_badge = "✅ COMPATIBLE WITH FD001 BASELINE" if curr_info.get("compatible") else "⚠️ INCOMPATIBLE (REQUIRES MODE 3)"
            badge_color = "#38bdf8" if curr_info.get("compatible") else "#fbbf24"
            st.markdown(
                f"<div class='card'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<b>{curr_info.get('name', sel_subset)}</b>"
                f"<span style='color:{badge_color}; font-weight:600; font-size:12px; border:1px solid {badge_color}; padding:2px 8px; border-radius:4px;'>{compat_badge}</span>"
                f"</div>"
                f"<div class='kpi-sub' style='margin-top:6px;'><b>Operating Regimes:</b> {curr_info.get('condition_desc')}</div>"
                f"<div class='kpi-sub'><b>Degradation Dynamics:</b> {curr_info.get('fault_desc')}</div>"
                f"<div class='kpi-sub'><b>Files:</b> Test: {curr_info.get('test_file') or 'N/A'} | RUL: {curr_info.get('rul_file') or 'N/A'} | Engines: {curr_info.get('test_engine_count', 0)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    rep = active["validation_report"]
    render_section_header("Dataset Validation Report", f"File: {rep['filename']} ({active['size_bytes'] / 1024:.1f} KB)")

    # Status summary
    col_status, col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(5)
    with col_status:
        st_color = {"VALID": "#38bdf8", "WARNING": "#fbbf24", "REJECTED": "#f87171"}.get(rep["status"], "#8d8c85")
        st.markdown(
            f"<div class='card'><div class='kpi-label'>Validation Status</div>"
            f"<div class='kpi-value' style='color:{st_color}'>{rep['status']}</div>"
            f"<div class='kpi-sub'>Quality gate</div></div>",
            unsafe_allow_html=True,
        )
    with col_kpi1:
        render_metric_card("Total Rows", f"{rep['row_count']:,}", sub=f"{rep['column_count']} columns")
    with col_kpi2:
        render_metric_card("Machines", f"{rep['machine_count']}", sub="Assets identified")
    with col_kpi3:
        render_metric_card("Avg Sequence", f"{rep['avg_sequence_length']:.0f}", sub=f"Min: {rep['min_sequence_length']} | Max: {rep['max_sequence_length']}")
    with col_kpi4:
        render_metric_card("Duplicates", f"{rep['duplicate_record_count']}", sub="Timestamp collisions")

    # Issues and warnings
    if rep["issues"]:
        st.error("**Validation Rejection Issues:**\n" + "\n".join(f"- {i}" for i in rep["issues"]))
    if rep["warnings"]:
        with st.expander(f"⚠️ Validation Warnings ({len(rep['warnings'])})", expanded=True):
            st.markdown("\n".join(f"- {w}" for w in rep["warnings"]))
            if rep["recommendations"]:
                st.markdown("**Actionable Recommendations:**\n" + "\n".join(f"- {r}" for r in rep["recommendations"]))

    # Model Safety Shield Check
    st.write("")
    render_section_header("Model Safety Shield: NASA C-MAPSS Compatibility Check")
    if rep["nasa_compatible"]:
        st.success(
            "✅ **NASA C-MAPSS Compatible**: This dataset contains all 17 expected turbofan features in compliant format. "
            "You can evaluate predictions using the NASA baseline model on the 'Model Safety Shield & Predict' tab."
        )
    else:
        st.markdown(
            f"<div style='border: 1px solid #f87171; background: rgba(248, 113, 113, 0.1); padding: 16px; border-radius: 8px; margin-bottom: 16px;'>"
            f"<h4 style='color: #f87171; margin-top:0;'>⚠️ Model Safety Shield Active: Incompatible Dataset</h4>"
            f"<p>{rep['nasa_compatibility_reason']}</p>"
            f"<p><b>Why is this blocked?</b> The NASA baseline neural network was trained exclusively on FD001 (single operating condition at sea level). "
            f"Blindly passing sensor readings from other machine categories (pumps, industrial motors) or multi-condition turbofan subsets (FD002–FD004) "
            f"will yield mathematically invalid, misleading, and dangerous RUL estimates ('garbage in, garbage out').</p>"
            f"<p><b>Required Action:</b> Use <b>Mode 3: Industry Adaptation</b> to map your sensors and adapt or train a dedicated "
            f"predictive maintenance model tailored to this machine category.</p></div>",
            unsafe_allow_html=True,
        )

    # Preview Data Columns
    with st.expander("Detected Column Schema & Data Profiles", expanded=False):
        col_table = {
            "Column": list(rep["column_profiles"].keys()),
            "Data Type": [p["dtype"] for p in rep["column_profiles"].values()],
            "Null Count": [p["null_count"] for p in rep["column_profiles"].values()],
            "Unique Values": [p["unique_count"] for p in rep["column_profiles"].values()],
            "Min": [p["min_value"] if p["min_value"] is not None else "-" for p in rep["column_profiles"].values()],
            "Mean": [f"{p['mean_value']:.2f}" if p["mean_value"] is not None else "-" for p in rep["column_profiles"].values()],
            "Max": [p["max_value"] if p["max_value"] is not None else "-" for p in rep["column_profiles"].values()],
        }
        st.markdown(html_table(col_table, max_height=350), unsafe_allow_html=True)



def render_mode2_predict(api: ApiClient):
    api = _ensure_client(api)
    render_page_header("EXTERNAL PREDICTION", "Predict Remaining Useful Life for compatible datasets")

    active = _get_active_dataset(api)
    if not active:
        st.info("Please upload an external CSV or ZIP dataset first in the 'Dataset Upload & Validation' tab.")
        return

    rep = active["validation_report"]
    if not rep["nasa_compatible"]:
        st.markdown(
            f"<div style='border: 1px solid #f87171; background: rgba(248, 113, 113, 0.1); padding: 16px; border-radius: 8px; margin-bottom: 16px;'>"
            f"<h4 style='color: #f87171; margin-top:0;'>⚠️ Prediction Blocked by Model Safety Shield</h4>"
            f"<p>{rep['nasa_compatibility_reason']}</p>"
            f"<p>To train or predict using a model built for this equipment or operating regime, navigate to <b>Mode 3: Industry Adaptation</b>.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("🚀 Proceed to Mode 3: Industry Adaptation", type="primary", key="go_to_mode3_btn"):
            st.session_state["operation_mode"] = "🏭 Mode 3: Industry Adaptation & Training"
            st.rerun()
        return

    st.success("✅ Dataset is verified compatible with the NASA baseline model. Select an asset to run RUL inference.")
    machine_ids = rep["machine_ids"]
    selected_machine = st.selectbox("Select Machine / Asset", machine_ids, index=0)

    # Check for Ground Truth RUL from uploaded archive
    zip_meta = active.get("zip_metadata", {})
    sub_info = zip_meta.get("subsets_info", {}).get(zip_meta.get("active_subset", ""), {})
    engine_ruls = sub_info.get("engine_ruls", {})
    true_rul = None
    try:
        true_rul = engine_ruls.get(int(selected_machine))
    except (ValueError, TypeError):
        pass

    if st.button("Run Prediction on Selected Asset", type="primary", key="mode2_run_pred_btn"):
        with st.spinner("Extracting time-series readings and running model inference..."):
            try:
                pred_res = api.predict_universal(
                    filepath=active["saved_path"],
                    machine_id=str(selected_machine),
                    model_id="nasa_cmapss_lstm_v1",
                )

                st.write("")
                render_section_header(f"Inference Results: Asset {selected_machine}")

                if true_rul is not None:
                    k1, k2, k3, k4 = st.columns(4)
                else:
                    k1, k2, k3 = st.columns(3)

                with k1:
                    render_metric_card(
                        "Predicted RUL",
                        f"{pred_res['predicted_rul']:.1f}",
                        sub=f"±{pred_res['uncertainty']:.1f} cycles ({pred_res['uncertainty_method'].split(';')[0]})",
                    )
                if true_rul is not None:
                    with k2:
                        err = abs(pred_res["predicted_rul"] - true_rul)
                        render_metric_card(
                            "Ground Truth RUL",
                            f"{true_rul}",
                            sub=f"Absolute Error: {err:.1f} cycles",
                        )
                with (k3 if true_rul is not None else k2):
                    render_metric_card(
                        "Cycles Observed",
                        f"{pred_res['cycles_used']}",
                        sub="Front-padded window" if pred_res.get("window_padded") else "Full window (30+ cycles)",
                    )
                with (k4 if true_rul is not None else k3):
                    render_metric_card(
                        "Model Architecture",
                        "NASA FD001 LSTM",
                        sub="17 channels normalized",
                    )

                # Explanation card
                st.markdown(
                    f"<div class='card' style='margin-top:14px;'>"
                    f"<div class='kpi-label'>Diagnostic Explanation</div>"
                    f"<p style='color:#f1f0ea; margin:6px 0 0;'>{pred_res['explanation']}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Sensor trends
                trends = pred_res.get("sensor_trends", [])
                if trends:
                    st.write("")
                    render_section_header("Sensor Movement Relative to Baseline")
                    trend_table = {
                        "Sensor Channel": [t["sensor"] for t in trends],
                        "Direction": [t["direction"].capitalize() for t in trends],
                        "Baseline Drift (σ units)": [f"{t['drift_std_units']:+.2f}σ" for t in trends],
                    }
                    st.markdown(html_table(trend_table, max_height=250), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")



def render_mode3_mapping(api: ApiClient):
    api = _ensure_client(api)
    render_page_header("INDUSTRY & SENSOR MAPPING", "Configure machine category and map external sensor channels")

    try:
        ind_data = api.list_industries()
    except ApiError as e:
        st.error(f"Could not load industry taxonomies: {str(e)}")
        return

    industries = ind_data["industries"]
    canonical = ind_data["canonical_sensors"]

    col1, col2 = st.columns(2)
    with col1:
        ind_names = [i["name"] for i in industries]
        sel_ind_name = st.selectbox("Target Industry", ind_names, index=2 if len(ind_names) > 2 else 0)
        sel_ind = next(i for i in industries if i["name"] == sel_ind_name)

    with col2:
        cat_names = [c["name"] for c in sel_ind["categories"]]
        sel_cat_name = st.selectbox("Machine Category", cat_names, index=0)
        sel_cat = next(c for c in sel_ind["categories"] if c["name"] == sel_cat_name)

    st.markdown(
        f"<div class='card'><div class='kpi-label'>{sel_ind['name']} ➔ {sel_cat['name']}</div>"
        f"<p style='color:#c3c2b7; margin:6px 0;'>{sel_cat['description']}</p>"
        f"<div class='kpi-sub'><b>Recommended Sensors:</b> {', '.join(sel_cat['recommended_sensors'])}</div>"
        f"<div class='kpi-sub'><b>Typical Failure Modes:</b> {', '.join(sel_cat['typical_failure_modes'])}</div></div>",
        unsafe_allow_html=True,
    )

    active = _get_active_dataset(api)
    if not active:
        st.info("Upload an external dataset (in Mode 2 or via sample button) to configure sensor mappings.")
        return

    rep = active["validation_report"]
    columns = rep["columns"]

    render_section_header("Configurable Sensor Mapping")
    st.markdown("Map your CSV columns to the internal canonical features required for model adaptation.")

    if st.button("Auto-Detect Mappings from Headers"):
        try:
            auto_cfg = api.auto_map_columns(columns, sel_cat["id"])
            st.session_state["sensor_mapping_config"] = auto_cfg
            st.success(f"Auto-mapped {len(auto_cfg['mappings'])} sensor channels based on column semantics.")
        except ApiError as e:
            st.error(f"Auto-mapping failed: {str(e)}")

    cfg = st.session_state.get("sensor_mapping_config")
    if not cfg:
        # Default auto map
        cfg = api.auto_map_columns(columns, sel_cat["id"])
        st.session_state["sensor_mapping_config"] = cfg

    # Display mappings table
    m_list = cfg.get("mappings", [])
    if m_list:
        map_table = {
            "Source CSV Column": [m["source_column"] for m in m_list],
            "Target Internal Feature": [m["target_feature"] for m in m_list],
            "Unit": [m.get("unit", "") for m in m_list],
            "Description": [m.get("description", "") for m in m_list],
        }
        st.markdown(html_table(map_table, max_height=300), unsafe_allow_html=True)
    else:
        st.warning("No sensors mapped yet. Click 'Auto-Detect Mappings from Headers'.")


def render_mode3_profiling(api: ApiClient):
    api = _ensure_client(api)
    render_page_header("DATASET PROFILING & CORRELATIONS", "Statistical distributions, outlier bounds, and sensor correlations")

    active = _get_active_dataset(api)
    if not active:
        st.info("Upload a dataset first to view data profiling and sensor correlations.")
        return

    rep = active["validation_report"]
    sensors = rep.get("sensor_candidates", [])
    if not sensors:
        st.warning("No numeric sensor columns found in dataset.")
        return

    selected_sensors = st.multiselect("Select sensors to profile", sensors, default=sensors[:8])
    if not selected_sensors:
        st.info("Select at least one sensor.")
        return

    with st.spinner("Computing statistical distributions and Pearson correlation matrix..."):
        try:
            profile = api.profile_dataset(
                filepath=active["saved_path"],
                sensor_columns=selected_sensors,
                machine_col=rep.get("selected_machine_col"),
            )
        except ApiError as e:
            st.error(f"Profiling error: {str(e)}")
            return

    # Distributions Table
    render_section_header("Sensor Distributions & Outlier Analysis")
    dists = profile["sensor_distributions"]
    dist_table = {
        "Sensor": [d["name"] for d in dists],
        "Mean": [f"{d['mean']:.3f}" for d in dists],
        "Std Dev": [f"{d['std']:.3f}" for d in dists],
        "Min": [f"{d['min']:.3f}" for d in dists],
        "Median": [f"{d['median']:.3f}" for d in dists],
        "Max": [f"{d['max']:.3f}" for d in dists],
        "Skewness": [f"{d['skewness']:.2f}" for d in dists],
        "3σ Outliers": [f"{d['outlier_count_3sigma']} ({d['outlier_percentage']}%)" for d in dists],
    }
    st.markdown(html_table(dist_table, max_height=320), unsafe_allow_html=True)

    # Correlation Heatmap
    if len(selected_sensors) > 1:
        render_section_header("Inter-Sensor Pearson Correlation Matrix")
        corr_matrix = profile["correlation_matrix"]
        s_names = profile["sensor_names"]
        z_vals = [[corr_matrix[s1].get(s2, 0.0) for s2 in s_names] for s1 in s_names]

        fig = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=s_names,
            y=s_names,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
        ))
        style_fig(fig, 420, "Sensors", "Sensors", title="Sensor Correlation Heatmap")
        st.plotly_chart(fig, width="stretch", key="corr_heatmap")


def render_mode3_training(api: ApiClient):
    api = _ensure_client(api)
    render_page_header("MODEL ADAPTATION TRAINING", "Train, validate, and register an adapted LSTM model on external data")

    active = _get_active_dataset(api)
    if not active:
        st.info("Upload a dataset first (e.g. synthetic pump failure dataset or custom machine CSV) before training.")
        return

    rep = active["validation_report"]
    cols = rep.get("columns", [])

    # Synchronize sensor mapping with active dataset to prevent stale column references
    active_path = active.get("saved_path", "")
    cfg = st.session_state.get("sensor_mapping_config")
    stale = False
    if cfg and isinstance(cfg, dict):
        m_col = cfg.get("machine_id_column")
        src_cols = [m.get("source_column") for m in cfg.get("mappings", [])]
        if (m_col and m_col not in cols) or any(sc not in cols for sc in src_cols):
            stale = True

    if not cfg or stale or st.session_state.get("mapped_dataset_path") != active_path:
        cfg = api.auto_map_columns(cols)
        st.session_state["sensor_mapping_config"] = cfg
        st.session_state["mapped_dataset_path"] = active_path

    # Dataset Alignment & Column Mapping Confirmation
    render_section_header("Dataset Alignment & Sensor Mapping", f"Source: {rep.get('filename')}")
    st.markdown("Confirm dataset schema alignment. Columns are auto-detected from your active file.")

    al_col1, al_col2, al_col3 = st.columns(3)
    with al_col1:
        # Machine ID column candidate
        default_m = rep.get("selected_machine_col")
        m_idx = cols.index(default_m) if default_m in cols else 0
        sel_machine_col = st.selectbox("Machine / Asset ID Column", cols, index=m_idx, key="train_sel_m_col")
    with al_col2:
        # Cycle column candidate
        default_c = rep.get("selected_cycle_col")
        c_idx = cols.index(default_c) if default_c in cols else min(1, len(cols) - 1)
        sel_cycle_col = st.selectbox("Time / Cycle Column", cols, index=c_idx, key="train_sel_c_col")
    with al_col3:
        # RUL column candidate
        rul_options = ["(Derive from Run-to-Failure degradation)"] + [c for c in cols if c not in {sel_machine_col, sel_cycle_col}]
        default_r = rep.get("selected_rul_col")
        r_idx = rul_options.index(default_r) if (default_r and default_r in rul_options) else 0
        sel_rul_choice = st.selectbox("RUL Ground-Truth Source", rul_options, index=r_idx, key="train_sel_r_col")
        sel_rul_col = None if sel_rul_choice.startswith("(") else sel_rul_choice

    # Select sensor feature columns
    sensor_candidates = [c for c in cols if c not in {sel_machine_col, sel_cycle_col, sel_rul_col}]
    # Default to available mapped sensors or first 8 candidate columns
    default_sensors = [m.get("source_column") for m in cfg.get("mappings", []) if m.get("source_column") in sensor_candidates]
    if not default_sensors:
        default_sensors = sensor_candidates[:8]

    sel_sensor_cols = st.multiselect("Sensor Columns for Model Adaptation", sensor_candidates, default=default_sensors, key="train_sel_sensors")
    if not sel_sensor_cols:
        st.warning("⚠️ Please select at least one sensor column to train the model.")
        return

    # Update active mapping configuration
    active_mappings = []
    for sc in sel_sensor_cols:
        existing = next((m for m in cfg.get("mappings", []) if m.get("source_column") == sc), None)
        target_name = existing["target_feature"] if existing else sc
        active_mappings.append({"source_column": sc, "target_feature": target_name})

    cfg["machine_id_column"] = sel_machine_col
    cfg["cycle_column"] = sel_cycle_col
    cfg["rul_column"] = sel_rul_col
    cfg["mappings"] = active_mappings
    st.session_state["sensor_mapping_config"] = cfg

    # Transparent Preprocessing Guide
    with st.expander("ℹ️ How Model Preprocessing & Prediction Works (Architecture Details)", expanded=False):
        st.markdown("""
        ### End-to-End Predictive Maintenance Pipeline
        1. **Asset-Level Partitioning (Zero Data Leakage):**
           - Assets/machines are partitioned 80% train / 20% validation at the unit ID level.
           - Time-series readings from validation machines are strictly withheld during training.
        2. **RUL Target Derivation:**
           - If an explicit RUL column is provided, it is used directly.
           - If no RUL column exists, run-to-failure RUL is derived automatically:
             $$\\text{RUL}(t) = \\min(\\text{Max Cycle} - t, \\text{RUL Clip})$$
        3. **StandardScaler Normalization:**
           - Mean $\\mu$ and standard deviation $\\sigma$ are computed strictly on training assets only to prevent data snooping.
        4. **Causal Sliding Windows:**
           - Converts sequential records into $L=30$ cycle rolling temporal tensors with zero future lookahead.
        5. **Recurrent Neural Network Training:**
           - Trains a 2-layer LSTM with Adam optimization, analytical BPTT, and early stopping on validation loss.
        6. **Universal Prediction & Uncertainty:**
           - Runs forward pass inference and Monte Carlo Dropout ($N=30$ stochastic samples) to yield $\\text{RUL} \\pm \\sigma$.
        """)

    render_section_header("Training Hyperparameters & Configuration")

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.text_input("Adapted Model Name", value=f"Adapted {rep['filename'][:15]} Model")
        industry = st.selectbox("Industry", ["Water & Wastewater", "Manufacturing", "Power Generation", "Aerospace"], index=0)
    with c2:
        machine_cat = st.selectbox("Machine Category", ["Centrifugal Pump", "Induction Motor", "Gas Turbine", "Air Compressor"], index=0)
        seq_len = st.slider("Sequence Window Length", min_value=10, max_value=60, value=30, step=5)
    with c3:
        epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=20, step=5)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)

    st.caption("Training enforces asset-level splitting (zero data leakage) and fits StandardScaler on training machines only.")

    if st.button("Launch Model Adaptation Training", type="primary", key="btn_train_model"):
        req_data = {
            "dataset_path": active["saved_path"],
            "mapping_config": cfg,
            "model_name": model_name,
            "industry": industry,
            "machine_category": machine_cat,
            "seq_len": seq_len,
            "rul_clip": 125,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.001,
            "hidden_units": 64,
            "val_fraction": 0.2,
            "patience": 6,
            "seed": 42,
            "run_in_background": True,
        }
        with st.spinner("Launching training job in background..."):
            try:
                res = api.train_model(req_data)
                job_id = res["job_id"]
                st.session_state["active_training_job_id"] = job_id
                st.success(f"Training job started: {job_id}")
            except ApiError as e:
                st.error(f"Failed to start training: {str(e)}")

    job_id = st.session_state.get("active_training_job_id")
    if job_id:
        render_section_header("Live Training Status", f"Job ID: {job_id}")
        prog_bar = st.progress(0)
        status_box = st.empty()

        # Poll status up to 30 times with 1s sleep
        for _ in range(40):
            try:
                stat = api.training_status(job_id)
                pct = stat.get("progress_pct", 0)
                prog_bar.progress(pct)
                status_box.markdown(
                    f"**Status:** `{stat['status'].upper()}` | **Epoch:** {stat['current_epoch']}/{stat['total_epochs']} | "
                    f"**Train Loss:** {stat['train_loss']:.4f} | **Val Loss:** {stat['val_loss']:.4f}"
                )
                if stat["status"] in ("completed", "failed"):
                    break
                time.sleep(1.0)
            except Exception:
                break

        if stat.get("status") == "completed":
            st.success(f"🎉 Training Completed Successfully! Model registered with ID: `{stat['model_id']}`")
            metrics = stat.get("metrics", {})
            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card("Validation RMSE", f"{metrics.get('rmse', 0.0):.2f}", sub="cycles")
            with m2:
                render_metric_card("Validation MAE", f"{metrics.get('mae', 0.0):.2f}", sub="cycles")
            with m3:
                render_metric_card("Validation R² Score", f"{metrics.get('r2', 0.0):.3f}", sub="coefficient of determination")
        elif stat.get("status") == "failed":
            st.error(f"Training failed: {stat.get('error_message')}")


def render_mode3_registry(api: ApiClient):
    api = _ensure_client(api)
    render_page_header("MODEL REGISTRY", "Catalog of active, benchmark, and adapted predictive maintenance models")

    try:
        data = api.list_models()
    except ApiError as e:
        st.error(f"Failed to load model registry: {str(e)}")
        return

    models = data.get("models", [])
    st.markdown(f"Total registered models: **{len(models)}**")

    # Models Comparison Table
    reg_table = {
        "Model ID": [m["model_id"] for m in models],
        "Name": [m["name"] for m in models],
        "Industry": [m["industry"] for m in models],
        "Machine Category": [m["machine_category"] for m in models],
        "Features": [f"{len(m['feature_schema'])} channels" for m in models],
        "RMSE (cycles)": [f"{m['evaluation_metrics'].get('rmse', '-'):.2f}" if isinstance(m['evaluation_metrics'].get('rmse'), (int, float)) else "-" for m in models],
        "MAE (cycles)": [f"{m['evaluation_metrics'].get('mae', '-'):.2f}" if isinstance(m['evaluation_metrics'].get('mae'), (int, float)) else "-" for m in models],
        "Status": [m["status"].upper() for m in models],
    }
    st.markdown(html_table(reg_table, max_height=320), unsafe_allow_html=True)

    # Detailed view of a selected model
    st.write("")
    render_section_header("Model Specification Details")
    mid_list = [m["model_id"] for m in models]
    chosen_mid = st.selectbox("Select Model", mid_list, index=0)
    meta = next(m for m in models if m["model_id"] == chosen_mid)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"- **Model ID:** `{meta['model_id']}`\n"
            f"- **Architecture:** {meta['architecture']}\n"
            f"- **Training Dataset:** {meta['training_dataset']}\n"
            f"- **Sequence Window:** {meta['sequence_length']} cycles\n"
            f"- **RUL Cap / Scale:** {meta['rul_clip']} cycles\n"
            f"- **Uncertainty:** {'MC-Dropout Supported' if meta['supports_uncertainty'] else 'Unavailable'}"
        )
    with c2:
        st.markdown(
            f"- **Feature Schema ({len(meta['feature_schema'])} channels):**\n"
            f"`{', '.join(meta['feature_schema'])}`\n\n"
            f"- **Evaluation Metrics:** {meta['evaluation_metrics']}\n"
            f"- **Description:** {meta['description']}"
        )


def render_mode3_predict(api: ApiClient):
    api = _ensure_client(api)
    render_page_header("UNIVERSAL RUL PREDICTION", "Predict Remaining Useful Life across any registered machine model")

    try:
        data = api.list_models()
        models = data.get("models", [])
    except ApiError as e:
        st.error(f"Failed to fetch model registry: {str(e)}")
        return

    active = _get_active_dataset(api)
    if not active:
        st.info("Upload an external dataset (e.g. Synthetic Pump dataset) to run predictions.")
        return

    rep = active["validation_report"]

    col1, col2 = st.columns(2)
    with col1:
        mod_choices = {f"{m['name']} ({m['model_id']})": m for m in models}
        sel_mod_label = st.selectbox("Select Inference Model", list(mod_choices.keys()), index=len(mod_choices) - 1)
        selected_model = mod_choices[sel_mod_label]

    with col2:
        machine_ids = rep.get("machine_ids", [])
        sel_machine = st.selectbox("Select Asset / Machine ID", machine_ids, index=0 if machine_ids else None)

    # Verify schema compatibility taking into account model's registered mapping and session mapping
    sensor_map: dict[str, str] = {}
    if selected_model.get("sensor_mapping"):
        sensor_map.update(selected_model["sensor_mapping"])

    session_map_cfg = st.session_state.get("sensor_mapping_config")
    if session_map_cfg and hasattr(session_map_cfg, "mappings"):
        for m in session_map_cfg.mappings:
            sensor_map[m.source_column] = m.target_feature
    elif isinstance(session_map_cfg, dict):
        for m in session_map_cfg.get("mappings", []):
            if isinstance(m, dict):
                sensor_map[m.get("source_column")] = m.get("target_feature")

    raw_cols = rep.get("columns", [])
    available_features = set(raw_cols)
    for src_col, tgt_feat in sensor_map.items():
        if src_col in raw_cols:
            available_features.add(tgt_feat)

    missing_feats = [f for f in selected_model["feature_schema"] if f not in available_features]
    if missing_feats:
        st.warning(
            f"⚠️ **Schema Mismatch:** Model '{selected_model['name']}' requires features `{missing_feats}` "
            f"which are not directly present in dataset columns `{raw_cols}`. "
            f"Please ensure sensors are mapped in Tab 1 (Industry & Sensor Mapping) or select a compatible model."
        )
    elif sensor_map:
        active_mappings = {s: t for s, t in sensor_map.items() if s in raw_cols}
        if active_mappings:
            st.info(f"✅ **Sensor Mapping Active:** Automatically translating dataset columns `{active_mappings}` to model features.")

    if st.button("Run Universal RUL Prediction", type="primary", disabled=bool(missing_feats)):
        # Extract last 30 rows for the machine
        from pathlib import Path
        import csv
        fpath = Path(active["saved_path"])
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                m_col = rep.get("selected_machine_col") or (
                    "motor_num" if "motor_num" in raw_cols else (
                        "pump_id" if "pump_id" in raw_cols else (
                            "unit" if "unit" in raw_cols else "asset_id"
                        )
                    )
                )
                asset_rows = [r for r in reader if r.get(m_col) == str(sel_machine)]
        else:
            asset_rows = []

        if not asset_rows:
            st.error("No readings found for selected asset.")
            return

        with st.spinner("Executing neural network forward pass with Monte-Carlo dropout uncertainty..."):
            try:
                # Format numeric readings
                formatted_readings = []
                for r in asset_rows[-50:]:
                    rec = {}
                    for k, v in r.items():
                        try:
                            rec[k] = float(v)
                        except (ValueError, TypeError):
                            rec[k] = v
                    formatted_readings.append(rec)

                res = api.predict_universal(
                    readings=formatted_readings,
                    model_id=selected_model["model_id"],
                    machine_id=str(sel_machine),
                    column_mapping=sensor_map if sensor_map else None,
                    mc_samples=30,
                )
                st.session_state["last_universal_prediction"] = res
            except ApiError as e:
                st.error(f"Prediction failed: {str(e)}")

    res = st.session_state.get("last_universal_prediction")
    if res:
        render_section_header("Prediction Results & Health Indicators", f"Asset: {res.get('machine_id')} | Model: {res.get('model_name')}")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric_card("Predicted RUL", f"{res['predicted_rul']:.1f}", unit="cycles", sub="Dominant indicator")
        with k2:
            unc_str = f"±{res['uncertainty']:.1f}" if res.get("uncertainty") is not None else "N/A"
            render_metric_card("Uncertainty", unc_str, unit="cycles", sub=res.get("uncertainty_method", "MC-Dropout"))
        with k3:
            render_metric_card("Cycles Observed", f"{res['cycles_used']}", sub="History window")
        with k4:
            pad_str = "Front-Padded" if res.get("window_padded") else "Full Window"
            render_metric_card("Window Status", pad_str, sub="30-cycle causal buffer")

        st.markdown(f"<div class='card'><div class='kpi-label'>Diagnostic Explanation</div>"
                    f"<div style='color:#e5e4dc; margin-top:4px;'>{res['explanation']}</div></div>", unsafe_allow_html=True)

        # Primary Sensor Trends Table
        trends = res.get("sensor_trends", [])
        if trends:
            render_section_header("Observed Sensor Trends Over Window")
            trend_table = {
                "Sensor Feature": [t["feature"] for t in trends],
                "Current Reading": [f"{t['current']:.3f}" for t in trends],
                "Window Delta": [f"{t['delta']:+.3f}" for t in trends],
                "Trend Direction": [t["direction"].upper() for t in trends],
            }
            st.markdown(html_table(trend_table, max_height=250), unsafe_allow_html=True)

    # Prediction History Table
    try:
        hist_data = api.prediction_history(limit=10)
        h_rows = hist_data.get("history", [])
        if h_rows:
            render_section_header("Recent Universal Prediction History")
            hist_table = {
                "Timestamp": [r["timestamp"][:19].replace("T", " ") for r in h_rows],
                "Model": [r["model_name"] for r in h_rows],
                "Machine ID": [r.get("machine_id", "-") for r in h_rows],
                "Predicted RUL": [f"{r['predicted_rul']:.1f} cycles" for r in h_rows],
                "Uncertainty": [f"±{r['uncertainty']:.1f}" if r.get("uncertainty") is not None else "-" for r in h_rows],
                "Window Padded": [str(r.get("window_padded", False)) for r in h_rows],
            }
            st.markdown(html_table(hist_table, max_height=260), unsafe_allow_html=True)
    except Exception:
        pass
