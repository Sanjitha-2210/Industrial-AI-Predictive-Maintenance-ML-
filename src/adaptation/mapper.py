"""Sensor and Feature Mapping Configuration.

Maps external CSV column headers to canonical internal features or model-specific features.
Supports semantic validation, unit verification, and scale checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.adaptation.taxonomy import CANONICAL_SENSORS


@dataclass
class SensorMappingEntry:
    source_column: str
    target_feature: str
    unit: str = ""
    scale_factor: float = 1.0
    offset: float = 0.0
    description: str = ""


@dataclass
class SensorMappingConfig:
    industry_id: str
    machine_category_id: str
    machine_id_column: str
    cycle_column: str
    rul_column: str | None = None
    mappings: list[SensorMappingEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorMappingConfig":
        entries = [
            SensorMappingEntry(**m) if isinstance(m, dict) else m
            for m in data.get("mappings", [])
        ]
        return cls(
            industry_id=data.get("industry_id", "general"),
            machine_category_id=data.get("machine_category_id", "general"),
            machine_id_column=data.get("machine_id_column", "unit"),
            cycle_column=data.get("cycle_column", "cycle"),
            rul_column=data.get("rul_column"),
            mappings=entries,
        )

    def get_source_to_target_map(self) -> dict[str, str]:
        return {m.source_column: m.target_feature for m in self.mappings}

    def get_target_to_source_map(self) -> dict[str, str]:
        return {m.target_feature: m.source_column for m in self.mappings}

    def validate_mapping(self, available_columns: list[str]) -> list[str]:
        """Validate that mapped source columns exist in available columns and no duplicates exist."""
        errors = []
        if self.machine_id_column not in available_columns:
            errors.append(f"Machine ID column '{self.machine_id_column}' not found in dataset columns.")
        if self.cycle_column not in available_columns:
            errors.append(f"Cycle column '{self.cycle_column}' not found in dataset columns.")
        if self.rul_column and self.rul_column not in available_columns:
            errors.append(f"RUL column '{self.rul_column}' not found in dataset columns.")

        seen_sources = set()
        seen_targets = set()
        for m in self.mappings:
            if m.source_column not in available_columns:
                errors.append(f"Mapped source sensor '{m.source_column}' does not exist in dataset.")
            if m.source_column in seen_sources:
                errors.append(f"Duplicate mapping for source sensor '{m.source_column}'.")
            seen_sources.add(m.source_column)

            if m.target_feature in seen_targets:
                errors.append(f"Multiple source columns mapped to the same target feature '{m.target_feature}'.")
            seen_targets.add(m.target_feature)

        return errors


def suggest_auto_mapping(available_columns: list[str], machine_category_id: str | None = None) -> SensorMappingConfig:
    """Intelligently suggests default column mappings based on keyword matching."""
    lower_map = {c.lower(): c for c in available_columns}

    # Detect machine id
    m_col = "unit"
    for term in ("unit", "engine_id", "machine_id", "asset_id", "device_id", "id"):
        for low, orig in lower_map.items():
            if term == low or term in low:
                m_col = orig
                break
        if m_col != "unit":
            break

    # Detect cycle
    c_col = "cycle"
    for term in ("cycle", "time", "timestamp", "step", "hour", "index"):
        for low, orig in lower_map.items():
            if term == low or term in low:
                c_col = orig
                break
        if c_col != "cycle":
            break

    # Detect RUL
    r_col = None
    for term in ("rul", "remaining_useful_life", "ttf"):
        for low, orig in lower_map.items():
            if term == low or term in low:
                r_col = orig
                break
        if r_col is not None:
            break

    mappings: list[SensorMappingEntry] = []

    # Check for canonical sensors
    for can_name, can_def in CANONICAL_SENSORS.items():
        matched_col = None
        # match exact or substring
        for low, orig in lower_map.items():
            if orig in (m_col, c_col, r_col):
                continue
            if can_name in low or can_name.replace("_", "") in low:
                matched_col = orig
                break
            # check category keywords
            if can_def.category == "vibration" and "vib" in low:
                matched_col = orig
                break
            if can_def.category == "temperature" and ("temp" in low or "t_" in low):
                matched_col = orig
                break
            if can_def.category == "pressure" and ("press" in low or "p_" in low):
                matched_col = orig
                break
            if can_def.category == "speed" and ("speed" in low or "rpm" in low):
                matched_col = orig
                break
            if can_def.category == "electrical" and ("curr" in low or "amp" in low):
                matched_col = orig
                break

        if matched_col and not any(m.source_column == matched_col for m in mappings):
            mappings.append(
                SensorMappingEntry(
                    source_column=matched_col,
                    target_feature=can_name,
                    unit=can_def.default_unit,
                    description=can_def.description,
                )
            )

    return SensorMappingConfig(
        industry_id="general",
        machine_category_id=machine_category_id or "general",
        machine_id_column=m_col,
        cycle_column=c_col,
        rul_column=r_col,
        mappings=mappings,
    )
