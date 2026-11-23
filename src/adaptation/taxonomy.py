"""Multi-industry taxonomy for predictive maintenance.

Defines target industry categories, machine types, standard canonical sensor channels,
units, and semantic requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SensorDefinition:
    name: str
    display_name: str
    category: str  # "temperature", "vibration", "pressure", "speed", "electrical", "fluid", "operational"
    default_unit: str
    description: str
    min_reasonable: float
    max_reasonable: float


@dataclass(frozen=True)
class MachineCategory:
    id: str
    name: str
    industry: str
    description: str
    recommended_sensors: list[str] = field(default_factory=list)
    typical_failure_modes: list[str] = field(default_factory=list)


# Standard canonical sensor definitions
CANONICAL_SENSORS: dict[str, SensorDefinition] = {
    "vibration_rms": SensorDefinition(
        name="vibration_rms",
        display_name="Vibration (RMS)",
        category="vibration",
        default_unit="mm/s",
        description="Overall vibration velocity RMS",
        min_reasonable=0.0,
        max_reasonable=100.0,
    ),
    "vibration_peak": SensorDefinition(
        name="vibration_peak",
        display_name="Peak Vibration",
        category="vibration",
        default_unit="g",
        description="Peak acceleration amplitude",
        min_reasonable=0.0,
        max_reasonable=50.0,
    ),
    "bearing_temp": SensorDefinition(
        name="bearing_temp",
        display_name="Bearing Temperature",
        category="temperature",
        default_unit="°C",
        description="Temperature at bearing housing",
        min_reasonable=-20.0,
        max_reasonable=200.0,
    ),
    "winding_temp": SensorDefinition(
        name="winding_temp",
        display_name="Winding Temperature",
        category="temperature",
        default_unit="°C",
        description="Stator winding temperature",
        min_reasonable=-20.0,
        max_reasonable=250.0,
    ),
    "discharge_pressure": SensorDefinition(
        name="discharge_pressure",
        display_name="Discharge Pressure",
        category="pressure",
        default_unit="bar",
        description="Outlet fluid pressure",
        min_reasonable=0.0,
        max_reasonable=400.0,
    ),
    "suction_pressure": SensorDefinition(
        name="suction_pressure",
        display_name="Suction Pressure",
        category="pressure",
        default_unit="bar",
        description="Inlet suction pressure",
        min_reasonable=0.0,
        max_reasonable=50.0,
    ),
    "motor_current": SensorDefinition(
        name="motor_current",
        display_name="Motor Current",
        category="electrical",
        default_unit="A",
        description="Phase operating current",
        min_reasonable=0.0,
        max_reasonable=1000.0,
    ),
    "motor_voltage": SensorDefinition(
        name="motor_voltage",
        display_name="Motor Voltage",
        category="electrical",
        default_unit="V",
        description="Supply voltage",
        min_reasonable=0.0,
        max_reasonable=10000.0,
    ),
    "rotational_speed": SensorDefinition(
        name="rotational_speed",
        display_name="Rotational Speed",
        category="speed",
        default_unit="rpm",
        description="Shaft angular velocity",
        min_reasonable=0.0,
        max_reasonable=50000.0,
    ),
    "flow_rate": SensorDefinition(
        name="flow_rate",
        display_name="Volumetric Flow Rate",
        category="fluid",
        default_unit="m³/h",
        description="Flow rate through system",
        min_reasonable=0.0,
        max_reasonable=5000.0,
    ),
    "oil_debris_count": SensorDefinition(
        name="oil_debris_count",
        display_name="Oil Debris Particle Count",
        category="fluid",
        default_unit="ppm",
        description="Metallic particle concentration in lube oil",
        min_reasonable=0.0,
        max_reasonable=2000.0,
    ),
}

# Supported industry taxonomies
INDUSTRIES: dict[str, dict[str, Any]] = {
    "aerospace": {
        "id": "aerospace",
        "name": "Aerospace & Aviation",
        "categories": {
            "turbofan": MachineCategory(
                id="turbofan",
                name="Turbofan Jet Engine",
                industry="Aerospace & Aviation",
                description="Commercial and military turbofan propulsion systems (e.g. C-MAPSS class).",
                recommended_sensors=["rotational_speed", "bearing_temp", "discharge_pressure"],
                typical_failure_modes=["HPC blade degradation", "Fan tip clearance rub", "Thermal barrier breakdown"],
            )
        },
    },
    "power_generation": {
        "id": "power_generation",
        "name": "Power Generation",
        "categories": {
            "gas_turbine": MachineCategory(
                id="gas_turbine",
                name="Industrial Gas Turbine",
                industry="Power Generation",
                description="Heavy-duty combustion turbines for grid power generation.",
                recommended_sensors=["rotational_speed", "bearing_temp", "discharge_pressure", "vibration_rms"],
                typical_failure_modes=["Compressor fouling", "Combustor transition wear", "Exhaust thermocouple drift"],
            ),
            "wind_turbine": MachineCategory(
                id="wind_turbine",
                name="Wind Turbine Gearbox & Generator",
                industry="Power Generation",
                description="Utility-scale wind turbine drive trains.",
                recommended_sensors=["rotational_speed", "vibration_rms", "bearing_temp", "oil_debris_count"],
                typical_failure_modes=["High-speed bearing spalling", "Planetary gear fatigue", "Generator winding overheating"],
            ),
        },
    },
    "water_and_wastewater": {
        "id": "water_and_wastewater",
        "name": "Water, Wastewater & Utilities",
        "categories": {
            "centrifugal_pump": MachineCategory(
                id="centrifugal_pump",
                name="Centrifugal Slurry / Water Pump",
                industry="Water, Wastewater & Utilities",
                description="High-duty industrial pumping equipment.",
                recommended_sensors=["vibration_rms", "bearing_temp", "suction_pressure", "discharge_pressure", "motor_current", "flow_rate"],
                typical_failure_modes=["Cavitation erosion", "Impeller imbalance", "Mechanical seal leakage", "Bearing raceway spalling"],
            )
        },
    },
    "manufacturing": {
        "id": "manufacturing",
        "name": "Manufacturing & Heavy Industrial",
        "categories": {
            "induction_motor": MachineCategory(
                id="induction_motor",
                name="Three-Phase Induction Motor",
                industry="Manufacturing & Heavy Industrial",
                description="AC induction motors driving conveyors, mills, and extruders.",
                recommended_sensors=["vibration_rms", "bearing_temp", "winding_temp", "motor_current", "rotational_speed"],
                typical_failure_modes=["Broken rotor bars", "Stator insulation degradation", "Bearing electrical fluting"],
            ),
            "air_compressor": MachineCategory(
                id="air_compressor",
                name="Rotary Screw Compressor",
                industry="Manufacturing & Heavy Industrial",
                description="Continuous compressed air plant systems.",
                recommended_sensors=["discharge_pressure", "bearing_temp", "vibration_rms", "motor_current"],
                typical_failure_modes=["Oil separator plugging", "Screw element wear", "Intake valve leakage"],
            ),
        },
    },
}


def get_all_industries() -> list[dict[str, Any]]:
    """Return all industries with their machine categories."""
    result = []
    for ind_id, ind_data in INDUSTRIES.items():
        cats = [
            {
                "id": cat.id,
                "name": cat.name,
                "description": cat.description,
                "recommended_sensors": cat.recommended_sensors,
                "typical_failure_modes": cat.typical_failure_modes,
            }
            for cat in ind_data["categories"].values()
        ]
        result.append({"id": ind_id, "name": ind_data["name"], "categories": cats})
    return result


def get_canonical_sensors_list() -> list[dict[str, Any]]:
    """Return the list of standard canonical sensors with metadata."""
    return [
        {
            "name": s.name,
            "display_name": s.display_name,
            "category": s.category,
            "default_unit": s.default_unit,
            "description": s.description,
            "min_reasonable": s.min_reasonable,
            "max_reasonable": s.max_reasonable,
        }
        for s in CANONICAL_SENSORS.values()
    ]
