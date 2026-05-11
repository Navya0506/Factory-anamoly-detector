"""
detector.py — Factory Floor Anomaly Detection Engine
=====================================================
Pure Python logic. No MQTT. No Docker. No external dependencies.

Implements 4 physics-based detectors:
  1. PWF — Power Failure         (torque × rpm → watts)
  2. TWF — Tool Wear Failure     (cumulative tool wear minutes)
  3. OSF — Overstrain Failure    (tool_wear × torque, Type-dependent)
  4. HDF — Heat Dissipation Failure (temp delta + rpm)

Priority when multiple fire simultaneously:
  PWF(1) > TWF(2) > OSF(3) > HDF(4) > RNF(5)

RNF (Random Failure) is intentionally undetectable — excluded from metrics.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

# Danger priority — lower number = more dangerous
DANGER_PRIORITY = {
    "PWF": 1,
    "TWF": 2,
    "OSF": 3,
    "HDF": 4,
    "RNF": 5,
}

# OSF strain index thresholds by machine Type
OSF_CRITICAL = {"L": 11000, "M": 12000, "H": 13000}
OSF_WARNING  = {"L":  9500, "M": 10500, "H": 11500}

# PWF safe power window (Watts)
PWF_CRITICAL_LOW  = 3500
PWF_CRITICAL_HIGH = 9000
PWF_WARNING_LOW   = 4000   # 500W buffer below overload floor
PWF_WARNING_HIGH  = 8000   # 1000W buffer below overload ceiling

# TWF tool wear thresholds (minutes)
TWF_WARNING  = 200
TWF_CRITICAL = 220

# HDF thermal thresholds
HDF_CRITICAL_DELTA = 8.6   # K — cooling collapse
HDF_WARNING_DELTA  = 9.5   # K — gradient weakening
HDF_CRITICAL_RPM   = 1380
HDF_WARNING_RPM    = 1400


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class SensorReading:
    """
    One row from the AI4I dataset, cleaned and typed.
    All units are as-is from the dataset.
    """
    product_id:       str
    machine_type:     str    # "L", "M", or "H"
    air_temp_k:       float  # Kelvin
    process_temp_k:   float  # Kelvin
    rpm:              float  # rotations per minute
    torque_nm:        float  # Newton-metres
    tool_wear_min:    float  # cumulative minutes
    actual_failure:   int    # 0 or 1 — used only for validation
    actual_twf:       int    # ground truth labels
    actual_hdf:       int
    actual_pwf:       int
    actual_osf:       int
    actual_rnf:       int


@dataclass
class DetectionResult:
    """
    Output from one detector for one sensor reading.
    """
    failure_type:  str            # "PWF", "TWF", "OSF", "HDF", "RNF"
    state:         str            # "NORMAL", "WARNING", "CRITICAL", "UNDETECTABLE"
    message:       str            # human-readable explanation
    sensor_values: dict           # the specific values that triggered this


@dataclass
class PipelineOutput:
    """
    Final output after all detectors run and priority engine resolves.
    This is what gets published to MQTT (later).
    """
    timestamp:        int
    product_id:       str
    machine_type:     str
    primary_alert:    Optional[DetectionResult]
    secondary_risks:  list[str]         = field(default_factory=list)
    all_results:      list[DetectionResult] = field(default_factory=list)
    actual_failure:   int               = 0
    early_warning:    bool              = False  # True if we warned before label = 1
    overall_state:    str               = "NORMAL"


# ─────────────────────────────────────────────
#  Individual Detectors
# ─────────────────────────────────────────────

def detect_pwf(reading: SensorReading) -> DetectionResult:
    """
    Power Failure Detector.
    Power (W) = torque (Nm) × angular_velocity (rad/s)
              = torque × (rpm × 2π / 60)
    Danger: power outside safe window [3500W, 9000W]
    Priority: 1 (most dangerous)
    """
    power_w = reading.torque_nm * (reading.rpm * 2 * math.pi / 60)
    power_w = round(power_w, 1)

    sensor_vals = {
        "power_watts": power_w,
        "torque_nm":   reading.torque_nm,
        "rpm":         reading.rpm,
    }

    # CRITICAL — outside hard limits
    if power_w <= PWF_CRITICAL_LOW:
        return DetectionResult(
            failure_type  = "PWF",
            state         = "CRITICAL",
            message       = (
                f"Power at {power_w}W — BELOW minimum {PWF_CRITICAL_LOW}W. "
                f"Motor stalling. Immediate shutdown recommended."
            ),
            sensor_values = sensor_vals,
        )
    if power_w >= PWF_CRITICAL_HIGH:
        return DetectionResult(
            failure_type  = "PWF",
            state         = "CRITICAL",
            message       = (
                f"Power at {power_w}W — ABOVE maximum {PWF_CRITICAL_HIGH}W. "
                f"Electrical overload imminent."
            ),
            sensor_values = sensor_vals,
        )

    # WARNING — approaching limits
    if power_w < PWF_WARNING_LOW:
        return DetectionResult(
            failure_type  = "PWF",
            state         = "WARNING",
            message       = (
                f"Power at {power_w}W — approaching stall floor of {PWF_CRITICAL_LOW}W. "
                f"Reduce cutting load."
            ),
            sensor_values = sensor_vals,
        )
    if power_w > PWF_WARNING_HIGH:
        return DetectionResult(
            failure_type  = "PWF",
            state         = "WARNING",
            message       = (
                f"Power at {power_w}W — approaching overload ceiling of {PWF_CRITICAL_HIGH}W. "
                f"Reduce feed rate."
            ),
            sensor_values = sensor_vals,
        )

    # NORMAL
    return DetectionResult(
        failure_type  = "PWF",
        state         = "NORMAL",
        message       = f"Power nominal at {power_w}W (safe window: {PWF_CRITICAL_LOW}–{PWF_CRITICAL_HIGH}W).",
        sensor_values = sensor_vals,
    )


def detect_twf(reading: SensorReading) -> DetectionResult:
    """
    Tool Wear Failure Detector.
    Tool fracture risk enters at 200 min, high risk at 220 min.
    Priority: 2
    """
    wear = reading.tool_wear_min

    sensor_vals = {
        "tool_wear_min":    wear,
        "warning_at_min":   TWF_WARNING,
        "critical_at_min":  TWF_CRITICAL,
    }

    if wear >= TWF_CRITICAL:
        return DetectionResult(
            failure_type  = "TWF",
            state         = "CRITICAL",
            message       = (
                f"Tool wear at {wear} min — past critical threshold of {TWF_CRITICAL} min. "
                f"Tool fracture imminent. Replace immediately."
            ),
            sensor_values = sensor_vals,
        )

    if wear >= TWF_WARNING:
        return DetectionResult(
            failure_type  = "TWF",
            state         = "WARNING",
            message       = (
                f"Tool wear at {wear} min — entered risk window ({TWF_WARNING}–{TWF_CRITICAL} min). "
                f"Schedule tool replacement."
            ),
            sensor_values = sensor_vals,
        )

    return DetectionResult(
        failure_type  = "TWF",
        state         = "NORMAL",
        message       = f"Tool wear nominal at {wear} min (risk window starts at {TWF_WARNING} min).",
        sensor_values = sensor_vals,
    )


def detect_osf(reading: SensorReading) -> DetectionResult:
    """
    Overstrain Failure Detector.
    Strain index = tool_wear_min × torque_nm
    Thresholds differ by machine Type (L / M / H).
    Priority: 3
    """
    machine_type = reading.machine_type.upper().strip()

    # Default to M if unknown type
    crit_threshold = OSF_CRITICAL.get(machine_type, OSF_CRITICAL["M"])
    warn_threshold = OSF_WARNING.get(machine_type,  OSF_WARNING["M"])

    strain_index = round(reading.tool_wear_min * reading.torque_nm, 1)

    sensor_vals = {
        "strain_index":     strain_index,
        "tool_wear_min":    reading.tool_wear_min,
        "torque_nm":        reading.torque_nm,
        "machine_type":     machine_type,
        "critical_at":      crit_threshold,
        "warning_at":       warn_threshold,
    }

    if strain_index >= crit_threshold:
        return DetectionResult(
            failure_type  = "OSF",
            state         = "CRITICAL",
            message       = (
                f"Strain index {strain_index} exceeds Type {machine_type} critical limit "
                f"of {crit_threshold}. Spindle overstrain — reduce torque or replace tool."
            ),
            sensor_values = sensor_vals,
        )

    if strain_index >= warn_threshold:
        return DetectionResult(
            failure_type  = "OSF",
            state         = "WARNING",
            message       = (
                f"Strain index {strain_index} approaching Type {machine_type} critical limit "
                f"of {crit_threshold} (warning at {warn_threshold}). Monitor closely."
            ),
            sensor_values = sensor_vals,
        )

    return DetectionResult(
        failure_type  = "OSF",
        state         = "NORMAL",
        message       = (
            f"Strain index nominal at {strain_index} "
            f"(Type {machine_type} limit: {crit_threshold})."
        ),
        sensor_values = sensor_vals,
    )


def detect_hdf(reading: SensorReading) -> DetectionResult:
    """
    Heat Dissipation Failure Detector.
    BOTH conditions must be true simultaneously:
      - temp delta below threshold
      - rpm below threshold
    Priority: 4
    """
    temp_delta = round(reading.process_temp_k - reading.air_temp_k, 2)
    rpm        = reading.rpm

    sensor_vals = {
        "temp_delta_k":    temp_delta,
        "process_temp_k":  reading.process_temp_k,
        "air_temp_k":      reading.air_temp_k,
        "rpm":             rpm,
    }

    # Both conditions must be true — this is the physics of HDF
    delta_critical = temp_delta < HDF_CRITICAL_DELTA
    rpm_critical   = rpm < HDF_CRITICAL_RPM
    delta_warning  = temp_delta < HDF_WARNING_DELTA
    rpm_warning    = rpm < HDF_WARNING_RPM

    if delta_critical and rpm_critical:
        return DetectionResult(
            failure_type  = "HDF",
            state         = "CRITICAL",
            message       = (
                f"Temp delta {temp_delta}K (min {HDF_CRITICAL_DELTA}K) AND "
                f"RPM {rpm} (min {HDF_CRITICAL_RPM}). "
                f"Cooling system failure — thermal runaway risk."
            ),
            sensor_values = sensor_vals,
        )

    if delta_warning and rpm_warning:
        return DetectionResult(
            failure_type  = "HDF",
            state         = "WARNING",
            message       = (
                f"Temp delta {temp_delta}K approaching minimum {HDF_CRITICAL_DELTA}K AND "
                f"RPM {rpm} below safe threshold {HDF_WARNING_RPM}. "
                f"Cooling gradient weakening."
            ),
            sensor_values = sensor_vals,
        )

    return DetectionResult(
        failure_type  = "HDF",
        state         = "NORMAL",
        message       = (
            f"Thermal nominal — delta {temp_delta}K, RPM {rpm}."
        ),
        sensor_values = sensor_vals,
    )


def detect_rnf() -> DetectionResult:
    """
    Random Failure — intentionally undetectable.
    Explicitly excluded from accuracy metrics.
    """
    return DetectionResult(
        failure_type  = "RNF",
        state         = "UNDETECTABLE",
        message       = (
            "Random failure — no sensor pattern exists. "
            "Excluded from accuracy metrics by design."
        ),
        sensor_values = {},
    )


# ─────────────────────────────────────────────
#  Priority Engine
# ─────────────────────────────────────────────

def resolve_priority(results: list[DetectionResult]) -> tuple[Optional[DetectionResult], list[str]]:
    """
    Given all detector results, find the most dangerous active alert.

    Returns:
        primary   — the single most dangerous WARNING or CRITICAL result
        secondary — names of other failure types also in WARNING/CRITICAL
    """
    # Filter to only active alerts (ignore NORMAL and UNDETECTABLE)
    active = [
        r for r in results
        if r.state in ("WARNING", "CRITICAL")
        and r.failure_type != "RNF"
    ]

    if not active:
        return None, []

    # Sort by danger priority (lowest number = most dangerous)
    active.sort(key=lambda r: DANGER_PRIORITY.get(r.failure_type, 99))

    primary   = active[0]
    secondary = [r.failure_type for r in active[1:]]

    return primary, secondary


# ─────────────────────────────────────────────
#  Main Pipeline Function
# ─────────────────────────────────────────────

def run_detectors(reading: SensorReading, timestamp: int = 0) -> PipelineOutput:
    """
    Run all 4 detectors on one sensor reading.
    Apply priority engine.
    Return structured PipelineOutput.

    This is the single function the publisher will call per CSV row.
    """

    # Run all detectors
    results = [
        detect_pwf(reading),
        detect_twf(reading),
        detect_osf(reading),
        detect_hdf(reading),
        detect_rnf(),
    ]

    # Resolve priority
    primary, secondary = resolve_priority(results)

    # Determine overall state
    if primary is None:
        overall_state = "NORMAL"
    elif primary.state == "CRITICAL":
        overall_state = "CRITICAL"
    else:
        overall_state = "WARNING"

    # Early warning flag:
    # True if we detected WARNING or CRITICAL but actual_failure is still 0
    # This is the proof that we caught it before the dataset's label fired
    early_warning = (overall_state in ("WARNING", "CRITICAL")) and (reading.actual_failure == 0)

    return PipelineOutput(
        timestamp      = timestamp,
        product_id     = reading.product_id,
        machine_type   = reading.machine_type,
        primary_alert  = primary,
        secondary_risks = secondary,
        all_results    = results,
        actual_failure = reading.actual_failure,
        early_warning  = early_warning,
        overall_state  = overall_state,
    )
