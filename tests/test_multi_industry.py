"""Comprehensive test suite for Multi-Industry Predictive Maintenance Platform.

Validates:
- Industry and machine taxonomies
- External dataset validation & error handling (missing columns, duplicates, sequence lengths)
- Model safety shield (rejecting blind inference on incompatible datasets)
- Sensor mapping engine (auto-mapping, manual configuration)
- Statistical profiling & Pearson correlation matrix
- Model Registry (registration, querying, persistence, schema compatibility checks)
- Model adaptation & training (asset-level split, scaling, metrics)
- Universal prediction engine (NASA benchmark and adapted models, MC-dropout uncertainty)
- FastAPI /api/v1 endpoints
- Security safeguards (file extensions, path traversal, size limits)
"""
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.api_v1 import (
    ProfileDatasetRequest,
    UniversalPredictRequest,
    ValidateDatasetRequest,
    list_industries,
    list_models,
    predict_universal,
    profile_dataset,
    validate_dataset,
)
from src import config as C
from src.adaptation.mapper import SensorMappingConfig, suggest_auto_mapping
from src.adaptation.profiler import compute_dataset_profile
from src.adaptation.taxonomy import CANONICAL_SENSORS, INDUSTRIES, get_all_industries, get_canonical_sensors_list
from src.adaptation.trainer import train_adapted_model
from src.adaptation.validator import read_csv_data, validate_csv_dataset
from src.data.table import from_records
from src.prediction_engine import IncompatibleSchemaError, get_prediction_engine
from src.registry.model_registry import ModelMetadata, get_model_registry
SAMPLE_DIR = C.ROOT / "data" / "sample_external"
PUMP_SAMPLE = SAMPLE_DIR / "synthetic_pump_run_to_failure.csv"
MOTOR_SAMPLE = SAMPLE_DIR / "synthetic_motor_incompatible.csv"


# ---- 1. Taxonomy Tests ----
def test_taxonomy_industries():
    industries = get_all_industries()
    assert len(industries) >= 4
    ind_ids = [i["id"] for i in industries]
    assert "aerospace" in ind_ids
    assert "water_and_wastewater" in ind_ids
    assert "manufacturing" in ind_ids

    # Check categories
    pump_cat = None
    for ind in industries:
        for cat in ind["categories"]:
            if cat["id"] == "centrifugal_pump":
                pump_cat = cat
                break
    assert pump_cat is not None
    assert "vibration_rms" in pump_cat["recommended_sensors"]


def test_taxonomy_canonical_sensors():
    sensors = get_canonical_sensors_list()
    assert len(sensors) >= 10
    names = [s["name"] for s in sensors]
    assert "vibration_rms" in names
    assert "bearing_temp" in names
    assert "motor_current" in names
    assert "discharge_pressure" in names


# ---- 2. Dataset Validation Tests ----
def test_validate_synthetic_pump_dataset():
    assert PUMP_SAMPLE.exists(), "Sample pump dataset should exist."
    report = validate_csv_dataset(PUMP_SAMPLE, filename="synthetic_pump_run_to_failure.csv")
    assert report.status in ("VALID", "WARNING")
    assert report.row_count > 100
    assert report.machine_count == 5
    assert "asset_id" in report.machine_id_candidates
    assert "cycle" in report.cycle_candidates
    assert "rul" in report.rul_candidates
    # NASA model compatibility should be False for pump data
    assert not report.nasa_compatible
    assert "Missing required NASA sensor features" in report.nasa_compatibility_reason


def test_validate_incompatible_motor_dataset():
    assert MOTOR_SAMPLE.exists(), "Sample motor dataset should exist."
    report = validate_csv_dataset(MOTOR_SAMPLE, filename="synthetic_motor_incompatible.csv")
    assert report.row_count > 0
    assert report.machine_count == 2
    assert not report.nasa_compatible


def test_validate_invalid_extension():
    report = validate_csv_dataset("fake_file.exe")
    assert report.status == "REJECTED"
    assert "Unsupported file extension" in report.issues[0]


def test_validate_empty_csv():
    report = validate_csv_dataset(b"")
    assert report.status == "REJECTED"


def test_validate_duplicates_and_non_monotonic():
    csv_content = (
        "unit,cycle,vibration,temp\n"
        "1,1,2.5,45.0\n"
        "1,2,2.6,45.2\n"
        "1,2,2.6,45.2\n"  # duplicate
        "1,1,2.7,45.3\n"  # non-monotonic
    ).encode("utf-8")
    report = validate_csv_dataset(csv_content, filename="test_dup.csv")
    assert report.duplicate_record_count > 0
    assert not report.has_monotonic_cycles


# ---- 3. Sensor Mapping Tests ----
def test_auto_mapping_pump_dataset():
    headers, _ = read_csv_data(PUMP_SAMPLE, max_rows=1)
    cfg = suggest_auto_mapping(headers, machine_category_id="centrifugal_pump")
    assert cfg.machine_id_column == "asset_id"
    assert cfg.cycle_column == "cycle"
    assert cfg.rul_column == "rul"
    mapped_targets = [m.target_feature for m in cfg.mappings]
    assert "vibration_rms" in mapped_targets
    assert "bearing_temp" in mapped_targets
    assert "motor_current" in mapped_targets


# ---- 4. Profiling Engine Tests ----
def test_profiling_engine():
    profile = compute_dataset_profile(
        PUMP_SAMPLE,
        sensor_columns=["vibration_rms", "bearing_temp", "motor_current", "discharge_pressure"],
        machine_col="asset_id",
    )
    assert profile.row_count > 0
    assert profile.machine_count == 5
    assert len(profile.sensor_distributions) == 4
    for d in profile.sensor_distributions:
        assert d.mean > 0
        assert d.std >= 0
    # Check correlation matrix
    assert "vibration_rms" in profile.correlation_matrix
    assert profile.correlation_matrix["vibration_rms"]["vibration_rms"] == 1.0


# ---- 5. Model Registry Tests ----
def test_model_registry_default_nasa():
    reg = get_model_registry()
    nasa_model = reg.get_model("nasa_cmapss_lstm_v1")
    assert nasa_model.name == "NASA C-MAPSS FD001 Turbofan LSTM Baseline"
    assert len(nasa_model.feature_schema) == 17
    assert nasa_model.sequence_length == 30
    assert nasa_model.status == "active"


def test_model_registry_compatibility_check():
    reg = get_model_registry()
    # Incompatible columns
    ok, reason, missing = reg.check_dataset_compatibility("nasa_cmapss_lstm_v1", ["cycle", "vibration", "temp"])
    assert not ok
    assert len(missing) > 0


# ---- 6. Model Training & Adaptation Tests ----
def test_train_adapted_model_pump():
    headers, _ = read_csv_data(PUMP_SAMPLE, max_rows=1)
    cfg = suggest_auto_mapping(headers, machine_category_id="centrifugal_pump")
    model_id, metrics = train_adapted_model(
        dataset_path=PUMP_SAMPLE,
        mapping_config=cfg,
        model_name="Test Pump LSTM",
        industry="Water & Wastewater",
        machine_category="Centrifugal Pump",
        seq_len=15,
        rul_clip=100,
        epochs=3,
        batch_size=32,
        seed=42,
    )
    assert model_id.startswith("model_centrifugal_pump_")
    assert "rmse" in metrics and metrics["rmse"] > 0
    assert "mae" in metrics and metrics["mae"] > 0
    assert "r2" in metrics

    # Verify model is now in registry
    reg = get_model_registry()
    meta = reg.get_model(model_id)
    assert meta.name == "Test Pump LSTM"
    assert meta.sequence_length == 15


# ---- 7. Prediction Engine Tests ----
def test_universal_prediction_engine():
    engine = get_prediction_engine()
    # Verify NASA model rejects incompatible schema
    incomp_table = {"cycle": [1, 2], "random_sensor": [1.0, 2.0]}
    with pytest.raises(IncompatibleSchemaError) as exc_info:
        engine.predict(incomp_table, model_id="nasa_cmapss_lstm_v1")
    assert "Missing required features" in str(exc_info.value)


# ---- 8. FastAPI Endpoints Tests ----
def test_api_v1_industries():
    data = list_industries()
    assert "industries" in data and "canonical_sensors" in data


def test_api_v1_models_registry():
    data = list_models()
    assert data["count"] >= 1
    m_ids = [m["model_id"] for m in data["models"]]
    assert "nasa_cmapss_lstm_v1" in m_ids


def test_api_v1_validate_dataset():
    data = validate_dataset(ValidateDatasetRequest(filepath="data/sample_external/synthetic_pump_run_to_failure.csv"))
    assert data["row_count"] > 100
    assert data["machine_count"] == 5


def test_api_v1_profile_dataset():
    data = profile_dataset(
        ProfileDatasetRequest(
            filepath="data/sample_external/synthetic_pump_run_to_failure.csv",
            sensor_columns=["vibration_rms", "bearing_temp"],
            machine_col="asset_id",
        )
    )
    assert "sensor_distributions" in data
    assert "correlation_matrix" in data


def test_api_v1_predict_reject_incompatible():
    readings = [{"cycle": 1, "unrelated_column": 12.5}]
    with pytest.raises(HTTPException) as exc_info:
        predict_universal(UniversalPredictRequest(model_id="nasa_cmapss_lstm_v1", readings=readings))
    assert exc_info.value.status_code == 422
    assert "Missing required features" in exc_info.value.detail


def test_api_v1_upload_dataset_endpoint():
    from io import BytesIO
    from fastapi import UploadFile
    from backend.api_v1 import upload_dataset

    content = b"machine_id,cycle,temp,pressure\nM1,1,65.2,101.3\nM1,2,65.5,101.4\n"
    uf = UploadFile(filename="unit_test_pump.csv", file=BytesIO(content))
    res = upload_dataset(uf)
    assert res["status"] == "success"
    assert "unit_test_pump.csv" in res["filename"]
    assert "saved_path" in res
    assert "validation_report" in res
    assert res["validation_report"]["row_count"] == 2
    assert "temp" in res["validation_report"]["columns"]

    # Clean up created upload file
    dest = C.ROOT / res["saved_path"]
    if dest.exists():
        dest.unlink(missing_ok=True)


def test_api_v1_upload_invalid_extension():
    from io import BytesIO
    from fastapi import UploadFile
    from backend.api_v1 import upload_dataset

    uf = UploadFile(filename="malicious.exe", file=BytesIO(b"MZ..."))
    with pytest.raises(HTTPException) as exc_info:
        upload_dataset(uf)
    assert exc_info.value.status_code == 400
    assert "Invalid file extension" in exc_info.value.detail


# ---- 9. ZIP Archive & NASA C-MAPSS Tests ----
def test_zip_valid_nasa_cmapss_upload():
    from io import BytesIO
    from fastapi import UploadFile
    from backend.api_v1 import upload_dataset

    zip_sample = C.ROOT / "data" / "sample_external" / "sample_cmapss_fd001_fd002.zip"
    assert zip_sample.exists(), "Sample C-MAPSS ZIP not found"
    with open(zip_sample, "rb") as f:
        data = f.read()

    uf = UploadFile(filename="cmapss_benchmark.zip", file=BytesIO(data))
    res = upload_dataset(uf)
    assert res["status"] == "success"
    assert "zip_metadata" in res
    zm = res["zip_metadata"]
    assert zm["is_zip"] is True
    assert zm["archive_type"] == "nasa_cmapss"
    assert "FD001" in zm["subsets_detected"]
    assert "FD002" in zm["subsets_detected"]
    assert zm["active_subset"] == "FD001"

    # Default active subset is FD001: must be compatible with baseline
    assert res["validation_report"]["nasa_compatible"] is True
    assert res["validation_report"]["row_count"] > 1000
    assert "sensor_2" in res["validation_report"]["columns"]


def test_zip_subset_switching_and_safety_shield():
    from backend.api_v1 import select_dataset_subset, SelectSubsetRequest

    # Select FD002 subset (which has 6 operating conditions)
    req = SelectSubsetRequest(
        subset="FD002",
        converted_csv_path="data/uploads/FD002_test_FD002.csv",
    )
    res = select_dataset_subset(req)
    assert res["status"] == "success"
    assert res["subset"] == "FD002"
    vr = res["validation_report"]
    assert vr["nasa_compatible"] is False
    assert "6 operating conditions" in vr["nasa_compatibility_reason"]
    assert "Mode 3" in vr["nasa_compatibility_reason"]


def test_zip_path_traversal_zip_slip_rejected():
    import zipfile
    from io import BytesIO
    from fastapi import UploadFile
    from backend.api_v1 import upload_dataset

    # Construct malicious archive containing Zip Slip path traversal
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, "w") as z:
        z.writestr("../../evil_payload.txt", "malicious_content")
    mem_zip.seek(0)

    uf = UploadFile(filename="attack.zip", file=mem_zip)
    with pytest.raises(HTTPException) as exc_info:
        upload_dataset(uf)
    assert exc_info.value.status_code == 400
    assert "traversal" in exc_info.value.detail.lower() or "security" in exc_info.value.detail.lower()


def test_zip_corrupted_archive_rejected():
    from io import BytesIO
    from fastapi import UploadFile
    from backend.api_v1 import upload_dataset

    corrupted_data = b"PK\x03\x04corrupted_header_data_not_a_valid_zip"
    uf = UploadFile(filename="broken.zip", file=BytesIO(corrupted_data))
    with pytest.raises(HTTPException) as exc_info:
        upload_dataset(uf)
    assert exc_info.value.status_code == 400
    assert "ZIP format error" in exc_info.value.detail or "Corrupted" in exc_info.value.detail


def test_zip_oversized_uncompressed_rejected():
    import zipfile
    from io import BytesIO
    from unittest.mock import patch
    from fastapi import UploadFile
    from backend.api_v1 import upload_dataset

    # Create zip and patch limit to verify 413 rejection
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, "w") as z:
        z.writestr("test.csv", b"col_a,col_b\n1,2\n3,4\n" * 100)
    mem_zip.seek(0)

    uf = UploadFile(filename="oversized.zip", file=mem_zip)
    with patch("src.adaptation.archive.MAX_UNCOMPRESSED_BYTES", 50):
        with pytest.raises(HTTPException) as exc_info:
            upload_dataset(uf)
    assert exc_info.value.status_code == 413
    assert "exceeds" in exc_info.value.detail.lower()



def test_zip_generic_csv_archive():
    import zipfile
    from io import BytesIO
    from fastapi import UploadFile
    from backend.api_v1 import upload_dataset

    # Archive containing generic industrial equipment CSV
    csv_bytes = b"pump_id,timestamp,vibration,temp\nP1,1,0.45,65.2\nP1,2,0.48,65.8\n"
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, "w") as z:
        z.writestr("industrial_telemetry/pump_feed.csv", csv_bytes)
    mem_zip.seek(0)

    uf = UploadFile(filename="pumps_data.zip", file=mem_zip)
    res = upload_dataset(uf)
    assert res["status"] == "success"
    zm = res["zip_metadata"]
    assert zm["is_zip"] is True
    assert zm["archive_type"] == "generic"
    assert res["validation_report"]["row_count"] == 2
    assert "vibration" in res["validation_report"]["columns"]
    # Model Safety shield should reject generic pump from NASA model
    assert res["validation_report"]["nasa_compatible"] is False


def test_zip_fd001_prediction_flow():
    from backend.api_v1 import predict_universal, UniversalPredictRequest

    req = UniversalPredictRequest(
        model_id="nasa_cmapss_lstm_v1",
        machine_id="34",
        filepath="data/uploads/FD001_test_FD001.csv",
    )
    res = predict_universal(req)
    assert abs(res["predicted_rul"] - 7.3) < 0.2
    assert res["uncertainty"] is not None
    assert res["cycles_used"] == 203
    assert res["model_id"] == "nasa_cmapss_lstm_v1"


def test_zip_matlab_archive_conversion():
    """Verifies that uploaded ZIP archives containing nested .zip and MATLAB .mat files are automatically converted."""
    user_zip = Path(r"C:\Users\sanji\.gemini\antigravity\brain\f7fb499f-f2a4-4183-a3c9-21dd7de4efee\.user_uploaded\media_1795465778627_bf904729.zip")
    if not user_zip.exists():
        pytest.skip("User-uploaded ZIP file not found in test environment.")

    from src.adaptation.archive import process_uploaded_zip
    report = process_uploaded_zip(user_zip, "1.+Algae+Raceway.zip")
    assert report.is_zip is True
    assert report.active_saved_path != ""
    assert report.validation_report["row_count"] >= 100
    assert "temperature" in report.validation_report["columns"]
    assert "density" in report.validation_report["columns"]




