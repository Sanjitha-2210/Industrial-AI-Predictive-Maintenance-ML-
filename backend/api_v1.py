"""FastAPI Router for Multi-Industry Predictive Maintenance Platform (v1).

Exposes endpoints for dataset upload, validation, profiling, sensor mapping,
industry taxonomies, model registry, model adaptation/training, and universal prediction.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src import config as C
from src.adaptation.mapper import SensorMappingConfig, suggest_auto_mapping
from src.adaptation.profiler import compute_dataset_profile
from src.adaptation.archive import (
    CMAPSS_SUBSETS,
    MAX_ZIP_UPLOAD_BYTES,
    ZipFormatError,
    ZipSecurityError,
    ZipSizeError,
    process_uploaded_zip,
)
from src.adaptation.taxonomy import get_all_industries, get_canonical_sensors_list
from src.adaptation.trainer import (
    TRAINING_JOBS,
    launch_training_background,
    train_adapted_model,
)
from src.adaptation.validator import (
    MAX_UPLOAD_BYTES,
    SAFE_EXTENSIONS,
    read_csv_data,
    validate_csv_dataset,
)
from src.data.table import from_records
from src.prediction_engine import IncompatibleSchemaError, get_prediction_engine
from src.registry.model_registry import get_model_registry


router = APIRouter(tags=["v1"])
UPLOAD_DIR = C.ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---- Pydantic Request Models ----
class ValidateDatasetRequest(BaseModel):
    filepath: Optional[str] = None
    filename: Optional[str] = "dataset.csv"
    machine_col: Optional[str] = None
    cycle_col: Optional[str] = None
    rul_col: Optional[str] = None
    required_seq_len: int = 30


class ProfileDatasetRequest(BaseModel):
    filepath: str
    sensor_columns: list[str]
    machine_col: Optional[str] = None


class AutoMapRequest(BaseModel):
    columns: list[str]
    machine_category_id: Optional[str] = None


class TrainModelRequest(BaseModel):
    dataset_path: str
    mapping_config: dict[str, Any]
    model_name: str
    industry: str
    machine_category: str
    seq_len: int = 30
    rul_clip: int = 125
    epochs: int = 25
    batch_size: int = 64
    lr: float = 0.001
    hidden_units: int = 64
    val_fraction: float = 0.2
    patience: int = 6
    seed: int = 42
    run_in_background: bool = True


class UniversalPredictRequest(BaseModel):
    model_id: str = "nasa_cmapss_lstm_v1"
    machine_id: Optional[str] = None
    readings: Optional[list[dict[str, Any]]] = None
    filepath: Optional[str] = None
    column_mapping: Optional[dict[str, str]] = None
    mc_samples: int = 30


class SelectSubsetRequest(BaseModel):
    subset: str
    converted_csv_path: str
    rul_file: Optional[str] = None


# ---- Endpoints ----
@router.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...)):
    """Uploads external CSV, TXT, or ZIP dataset securely with extension, size, and traversal checks."""
    orig_name = Path(file.filename or "upload.csv").name
    ext = Path(orig_name).suffix.lower()
    if ext not in SAFE_EXTENSIONS:
        raise HTTPException(
            400,
            detail=f"Invalid file extension '{ext}'. Supported formats: .csv, .txt, .zip, .tsv, .dat, .mat",
        )

    max_bytes = MAX_ZIP_UPLOAD_BYTES if ext == ".zip" else MAX_UPLOAD_BYTES
    safe_filename = f"{uuid.uuid4().hex[:8]}_{orig_name}"
    dest_path = UPLOAD_DIR / safe_filename

    # Save safely via synchronous underlying file reader
    bytes_written = 0
    with open(dest_path, "wb") as buffer:
        while chunk := file.file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                buffer.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    detail=f"File exceeds maximum upload size of {max_bytes // (1024*1024)} MB.",
                )
            buffer.write(chunk)

    if ext == ".zip":
        try:
            zip_report = process_uploaded_zip(dest_path, orig_name, output_base_dir=UPLOAD_DIR)
            return {
                "status": "success",
                "filename": orig_name,
                "saved_path": zip_report.active_saved_path,
                "size_bytes": bytes_written,
                "validation_report": zip_report.validation_report,
                "zip_metadata": {
                    "is_zip": True,
                    "archive_type": zip_report.archive_type,
                    "subsets_detected": zip_report.subsets_detected,
                    "active_subset": zip_report.active_subset,
                    "subsets_info": zip_report.subsets_info,
                    "extracted_file_count": zip_report.extracted_file_count,
                    "total_uncompressed_bytes": zip_report.total_uncompressed_bytes,
                },
            }
        except ZipSecurityError as e:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(400, detail=f"ZIP security violation: {str(e)}")
        except ZipSizeError as e:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(413, detail=f"ZIP size limit exceeded: {str(e)}")
        except ZipFormatError as e:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(400, detail=f"ZIP format error: {str(e)}")
        except Exception as e:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(500, detail=f"Failed to process ZIP archive: {str(e)}")

    if ext == ".mat":
        from src.adaptation.archive import convert_mat_to_csv
        conv_csv = UPLOAD_DIR / f"converted_{Path(safe_filename).stem}.csv"
        res_csv = convert_mat_to_csv(dest_path, conv_csv)
        if res_csv and res_csv.exists():
            dest_path = res_csv
            orig_name = f"{Path(orig_name).stem}.csv"
        else:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(400, detail="Could not extract 2D tabular time-series from MATLAB .mat file.")

    # Initial fast validation for standard CSV/TXT/TSV/DAT
    report = validate_csv_dataset(dest_path, filename=orig_name)
    return {
        "status": "success",
        "filename": orig_name,
        "saved_path": str(dest_path.relative_to(C.ROOT)).replace("\\", "/"),
        "size_bytes": bytes_written,
        "validation_report": report.to_dict(),
        "zip_metadata": {"is_zip": False},
    }


@router.post("/datasets/select-subset")
def select_dataset_subset(req: SelectSubsetRequest):
    """Switches the active subset from an extracted NASA C-MAPSS archive and updates validation."""
    target = C.ROOT / req.converted_csv_path
    if not target.exists():
        raise HTTPException(404, detail=f"Converted subset CSV not found: {req.converted_csv_path}")

    subset_key = req.subset.upper()
    if subset_key not in CMAPSS_SUBSETS:
        raise HTTPException(400, detail=f"Unknown NASA C-MAPSS subset: '{req.subset}'. Expected FD001-FD004.")

    spec = CMAPSS_SUBSETS[subset_key]
    report = validate_csv_dataset(target, filename=target.name)
    report.nasa_compatible = spec["compatible"]
    report.nasa_compatibility_reason = spec["reason"]

    return {
        "status": "success",
        "subset": subset_key,
        "saved_path": req.converted_csv_path,
        "validation_report": report.to_dict(),
    }




@router.post("/datasets/validate")
def validate_dataset(req: ValidateDatasetRequest):
    """Deep validation of an uploaded CSV dataset."""
    if not req.filepath:
        raise HTTPException(422, detail="Provide 'filepath' of the dataset to validate.")

    target = C.ROOT / req.filepath
    if not target.exists():
        raise HTTPException(404, detail=f"File not found: {req.filepath}")

    report = validate_csv_dataset(
        target,
        filename=req.filename or target.name,
        machine_col=req.machine_col,
        cycle_col=req.cycle_col,
        rul_col=req.rul_col,
        required_seq_len=req.required_seq_len,
    )
    return report.to_dict()


@router.post("/datasets/profile")
def profile_dataset(req: ProfileDatasetRequest):
    """Computes statistical distributions and sensor correlation matrix."""
    target = C.ROOT / req.filepath
    if not target.exists():
        raise HTTPException(404, detail=f"File not found: {req.filepath}")

    try:
        profile = compute_dataset_profile(
            target,
            sensor_columns=req.sensor_columns,
            machine_col=req.machine_col,
        )
        return profile.to_dict()
    except Exception as e:
        raise HTTPException(422, detail=f"Dataset profiling failed: {str(e)}")


@router.post("/datasets/auto-map")
def auto_map_columns(req: AutoMapRequest):
    """Suggests intelligent sensor mapping from column headers."""
    cfg = suggest_auto_mapping(req.columns, req.machine_category_id)
    return cfg.to_dict()


@router.get("/industries")
def list_industries():
    """Lists supported industry domains, machine categories, and canonical sensor definitions."""
    return {
        "industries": get_all_industries(),
        "canonical_sensors": get_canonical_sensors_list(),
    }


@router.get("/models/registry")
def list_models(
    industry: Optional[str] = None,
    machine_category: Optional[str] = None,
    status: Optional[str] = None,
):
    """Lists all registered models from the model registry."""
    reg = get_model_registry()
    models = reg.list_models(industry=industry, machine_category=machine_category, status=status)
    return {"count": len(models), "models": [m.to_dict() for m in models]}


@router.get("/models/{model_id}")
def get_model_details(model_id: str):
    """Returns metadata and metrics for a specific model."""
    reg = get_model_registry()
    try:
        meta = reg.get_model(model_id)
        return meta.to_dict()
    except KeyError:
        raise HTTPException(404, detail=f"Model '{model_id}' not found in registry.")


@router.post("/models/train")
def train_model(req: TrainModelRequest):
    """Starts model adaptation on an external CSV dataset."""
    target = C.ROOT / req.dataset_path
    if not target.exists():
        raise HTTPException(404, detail=f"Dataset not found: {req.dataset_path}")

    mapping_cfg = SensorMappingConfig.from_dict(req.mapping_config)

    if req.run_in_background:
        job_id = launch_training_background(
            dataset_path=target,
            mapping_config=mapping_cfg,
            model_name=req.model_name,
            industry=req.industry,
            machine_category=req.machine_category,
            seq_len=req.seq_len,
            rul_clip=req.rul_clip,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            hidden_units=req.hidden_units,
            val_fraction=req.val_fraction,
            patience=req.patience,
            seed=req.seed,
        )
        return {
            "status": "training_started",
            "job_id": job_id,
            "message": f"Training job '{job_id}' started in background.",
        }
    else:
        try:
            model_id, metrics = train_adapted_model(
                dataset_path=target,
                mapping_config=mapping_cfg,
                model_name=req.model_name,
                industry=req.industry,
                machine_category=req.machine_category,
                seq_len=req.seq_len,
                rul_clip=req.rul_clip,
                epochs=req.epochs,
                batch_size=req.batch_size,
                lr=req.lr,
                hidden_units=req.hidden_units,
                val_fraction=req.val_fraction,
                patience=req.patience,
                seed=req.seed,
            )
            return {
                "status": "completed",
                "model_id": model_id,
                "metrics": metrics,
            }
        except Exception as e:
            raise HTTPException(422, detail=f"Training failed: {str(e)}")


@router.get("/models/training/{job_id}")
def get_training_status(job_id: str):
    """Returns the status and progress of a background training job."""
    if job_id not in TRAINING_JOBS:
        raise HTTPException(404, detail=f"Training job '{job_id}' not found.")
    return TRAINING_JOBS[job_id].to_dict()


@router.post("/predict")
def predict_universal(req: UniversalPredictRequest):
    """Universal multi-model prediction endpoint with schema validation."""
    engine = get_prediction_engine()

    readings = req.readings
    if not readings:
        if req.filepath and req.machine_id:
            target = C.ROOT / req.filepath
            if not target.exists():
                raise HTTPException(404, detail=f"Dataset file not found: {req.filepath}")
            header, rows = read_csv_data(target)
            m_col = "unit" if "unit" in header else ("machine_id" if "machine_id" in header else header[0])
            m_idx = header.index(m_col)
            target_mid = str(req.machine_id)
            readings = []
            for r in rows:
                if str(r[m_idx]) == target_mid:
                    row_dict = {}
                    for k, v in zip(header, r):
                        try:
                            row_dict[k] = float(v)
                        except (ValueError, TypeError):
                            row_dict[k] = v
                    readings.append(row_dict)
            if not readings:
                raise HTTPException(404, detail=f"Machine '{req.machine_id}' not found in dataset '{req.filepath}'.")

        else:
            raise HTTPException(422, detail="Provide 'readings' array or both 'filepath' and 'machine_id'.")

    # Convert readings to Table
    try:
        table = from_records(readings)
    except Exception as e:
        raise HTTPException(422, detail=f"Malformed readings format: {str(e)}")

    try:
        res = engine.predict(
            table=table,
            model_id=req.model_id,
            machine_id=req.machine_id,
            column_mapping=req.column_mapping,
            mc_samples=req.mc_samples,
        )
        return res.to_dict()
    except IncompatibleSchemaError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction error: {str(e)}")



@router.get("/predictions/history")
def prediction_history(limit: int = Query(50, ge=1, le=200)):
    """Returns recent prediction history."""
    engine = get_prediction_engine()
    return {"history": engine.get_history(limit=limit)}
