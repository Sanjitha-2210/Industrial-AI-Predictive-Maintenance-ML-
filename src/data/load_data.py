"""Load NASA C-MAPSS files (NumPy only) with clear errors when the dataset is missing."""
from pathlib import Path

import numpy as np

from src.config import DATASET, RAW_DIR
from src.data.table import Table

OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS


class DatasetMissingError(FileNotFoundError):
    pass


def _path(prefix: str, raw_dir: Path, dataset: str) -> Path:
    p = Path(raw_dir) / f"{prefix}_{dataset}.txt"
    if not p.exists():
        raise DatasetMissingError(
            f"Missing {p}.\n"
            f"Download the NASA C-MAPSS (Turbofan Engine Degradation Simulation) dataset from the NASA "
            f"Prognostics Data Repository and place train_{dataset}.txt, test_{dataset}.txt and "
            f"RUL_{dataset}.txt in {Path(raw_dir)}."
        )
    return p


def _read_table(path: Path) -> Table:
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != len(COLUMNS):
        raise ValueError(f"{path} has {a.shape[1]} columns, expected {len(COLUMNS)}.")
    t = {c: a[:, i] for i, c in enumerate(COLUMNS)}
    t["unit"] = t["unit"].astype(np.int64)
    t["cycle"] = t["cycle"].astype(np.int64)
    return t


def load_train(raw_dir: Path = RAW_DIR, dataset: str = DATASET) -> Table:
    return _read_table(_path("train", raw_dir, dataset))


def load_test(raw_dir: Path = RAW_DIR, dataset: str = DATASET) -> Table:
    return _read_table(_path("test", raw_dir, dataset))


def load_test_rul(raw_dir: Path = RAW_DIR, dataset: str = DATASET) -> dict[int, int]:
    """True RUL at the last recorded cycle of each test engine, keyed by engine id (1-based)."""
    values = np.loadtxt(_path("RUL", raw_dir, dataset), ndmin=1).astype(int)
    return {i + 1: int(v) for i, v in enumerate(values)}


def dataset_available(raw_dir: Path = RAW_DIR, dataset: str = DATASET) -> bool:
    return all((Path(raw_dir) / f"{p}_{dataset}.txt").exists() for p in ("train", "test", "RUL"))
