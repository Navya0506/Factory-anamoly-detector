"""
test_detector.py — Validate detection logic against AI4I 2020 dataset
======================================================================
Run this BEFORE adding MQTT or Docker.
Purpose: prove the detector catches real failures and issues early warnings.

Usage:
    python test_detector.py                        # uses ai4i2020.csv in ../data/
    python test_detector.py --path /your/path.csv  # custom path
    python test_detector.py --rows 500             # test first N rows only

What this tests:
    1. Basic sanity — does each detector run without errors
    2. Early warning rate — how often do we warn BEFORE actual_failure = 1
    3. Per-failure-type accuracy — does PWF detector catch PWF rows etc.
    4. False positive rate — how often do we alert when nothing is actually wrong
    5. Priority engine — does it pick the right failure when multiple fire
"""

import sys
import os
import argparse
import time
import pandas as pd

# Add parent directory so we can import detector.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import SensorReading, run_detectors, DANGER_PRIORITY

# ─────────────────────────────────────────────
#  CSV → SensorReading
# ─────────────────────────────────────────────

def parse_row(row) -> SensorReading:
    """
    Convert one pandas DataFrame row into a SensorReading.
    Handles the AI4I column naming format.
    """
    return SensorReading(
        product_id     = str(row.get("product id",  row.get("product_id", "unknown"))),
        machine_type   = str(row.get("type",        "M")).strip().upper(),
        air_temp_k     = float(row.get("air temperature [k]",     row.get("air_temperature_[k]",     300.0))),
        process_temp_k = float(row.get("process temperature [k]", row.get("process_temperature_[k]", 310.0))),
        rpm            = float(row.get("rotational speed [rpm]",  row.get("rotational_speed_[rpm]",  1500.0))),
        torque_nm      = float(row.get("torque [nm]",             row.get("torque_[nm]",             40.0))),
        tool_wear_min  = float(row.get("tool wear [min]",         row.get("tool_wear_[min]",         0.0))),
        actual_failure = int(row.get("machine failure",           row.get("machine_failure",         0))),
        actual_twf     = int(row.get("twf", 0)),
        actual_hdf     = int(row.get("hdf", 0)),
        actual_pwf     = int(row.get("pwf", 0)),
        actual_osf     = int(row.get("osf", 0)),
        actual_rnf     = int(row.get("rnf", 0)),
    )

# ─────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────

STATE_ICON = {
    "NORMAL":        "🟢",
    "WARNING":       "🟡",
    "CRITICAL":      "🔴",
    "UNDETECTABLE":  "⚪",
}

FAILURE_COLOUR = {
    "PWF": "\033[91m",   # red
    "TWF": "\033[91m",   # red
    "OSF": "\033[93m",   # yellow
    "HDF": "\033[93m",   # yellow
    "RNF": "\033[90m",   # grey
}
RESET = "\033[0m"

def print_alert(output, row_num: int):
    """Print a single row result — only for non-NORMAL rows."""
    p = output.primary_alert
    icon  = STATE_ICON.get(output.overall_state, "❓")
    color = FAILURE_COLOUR.get(p.failure_type, "") if p else ""

    print(f"\n{'─'*60}")
    print(f"Row {row_num:5d} | {icon} {output.overall_state} | "
          f"Product: {output.product_id} | Type: {output.machine_type}")

    if p:
        print(f"  {color}▶ PRIMARY [{p.failure_type}] {p.state}{RESET}")
        print(f"    {p.message}")
        print(f"    Sensors: {p.sensor_values}")

    if output.secondary_risks:
        print(f"  ⚠  Secondary risks also active: {', '.join(output.secondary_risks)}")

    actual_flags = []
    if output.actual_failure:   actual_flags.append("FAILURE")
    print(f"  Dataset label: {'  '.join(actual_flags) if actual_flags else 'no failure'} "
          f"| Early warning: {'✅ YES' if output.early_warning else 'no'}")


# ─────────────────────────────────────────────
#  Metrics tracking
# ─────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.total_rows       = 0
        self.total_failures   = 0   # rows where actual_failure = 1
        self.detected         = 0   # we warned AND actual_failure = 1
        self.early_warnings   = 0   # we warned BUT actual_failure = 0
        self.false_positives  = 0   # we said CRITICAL but actual = 0
        self.missed           = 0   # actual_failure = 1 but we said NORMAL

        # Per failure type: did our specific detector catch its specific failure?
        self.per_type = {
            t: {"caught": 0, "total": 0}
            for t in ["PWF", "TWF", "OSF", "HDF", "RNF"]
        }

        self.priority_resolutions = 0  # how many times >1 detector fired

    def update(self, output, reading: SensorReading):
        self.total_rows += 1
        alerted = output.overall_state in ("WARNING", "CRITICAL")

        if reading.actual_failure:
            self.total_failures += 1
            if alerted:
                self.detected += 1
            else:
                self.missed += 1
        else:
            if alerted:
                self.early_warnings += 1
                if output.overall_state == "CRITICAL":
                    self.false_positives += 1

        if len(output.secondary_risks) > 0:
            self.priority_resolutions += 1

        # Per-type accuracy (only for detectable types)
        for t in ["PWF", "TWF", "OSF", "HDF"]:
            actual_flag = getattr(reading, f"actual_{t.lower()}", 0)
            if actual_flag:
                self.per_type[t]["total"] += 1
                # Check if our detector for this type fired
                detector_result = next(
                    (r for r in output.all_results if r.failure_type == t), None
                )
                if detector_result and detector_result.state in ("WARNING", "CRITICAL"):
                    self.per_type[t]["caught"] += 1

    def print_summary(self):
        print(f"\n{'═'*60}")
        print("  DETECTION RESULTS SUMMARY")
        print(f"{'═'*60}")
        print(f"  Total rows processed     : {self.total_rows:,}")
        print(f"  Actual failures in data  : {self.total_failures:,} "
              f"({100*self.total_failures/max(self.total_rows,1):.1f}%)")
        print()
        print(f"  ✅ Failures detected     : {self.detected} / {self.total_failures} "
              f"({100*self.detected/max(self.total_failures,1):.1f}%)")
        print(f"  🟡 Early warnings issued : {self.early_warnings} "
              f"(alerted before label=1)")
        print(f"  ❌ Failures missed       : {self.missed} "
              f"(actual=1 but we said NORMAL)")
        print(f"  ⚠  False positives (CRIT): {self.false_positives} "
              f"(we said CRITICAL, dataset said OK)")
        print(f"  🔀 Priority resolutions  : {self.priority_resolutions} "
              f"(multiple detectors fired)")
        print()
        print("  Per-failure-type detection rate:")
        for t, counts in self.per_type.items():
            if counts["total"] == 0:
                print(f"    {t}: no actual instances in tested rows")
            else:
                rate = 100 * counts["caught"] / counts["total"]
                bar_len = int(rate / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    {t}: [{bar}] {rate:.0f}%  "
                      f"({counts['caught']}/{counts['total']} caught)")
        print(f"{'═'*60}\n")


# ─────────────────────────────────────────────
#  Main test runner
# ─────────────────────────────────────────────

def run_tests(csv_path: str, max_rows: int = None, verbose: bool = True):
    print(f"\n{'═'*60}")
    print("  FACTORY FLOOR ANOMALY DETECTOR — TEST RUN")
    print(f"{'═'*60}")
    print(f"  Dataset : {csv_path}")
    print(f"  Rows    : {'all' if max_rows is None else max_rows}")
    print()

    # Load dataset
    try:
        df = pd.read_csv(r"C:\Navya\Project 2026\Factory anamoly detection\ai4i2020.csv")
    except FileNotFoundError:
        print(f"❌  File not found: {csv_path}")
        print("    Download from: https://www.kaggle.com/datasets/stephanmatzka/"
              "predictive-maintenance-dataset-ai4i-2020")
        print("    Save as: data/ai4i2020.csv")
        sys.exit(1)

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()
    print(f"  Loaded {len(df):,} rows | columns: {list(df.columns)[:6]} ...")

    if max_rows:
        df = df.head(max_rows)
        print(f"  Testing first {max_rows} rows only")

    metrics  = Metrics()
    start_ts = int(time.time())

    for i, (_, row) in enumerate(df.iterrows()):
        reading   = parse_row(row)
        timestamp = start_ts + i
        output    = run_detectors(reading, timestamp)

        metrics.update(output, reading)

        # Print alerts — only non-normal rows to keep output readable
        if verbose and output.overall_state != "NORMAL":
            print_alert(output, i + 1)

    metrics.print_summary()


# ─────────────────────────────────────────────
#  Quick sanity tests (no CSV needed)
# ─────────────────────────────────────────────

def run_sanity_checks():
    """
    Test each detector in isolation with hand-crafted values.
    Confirms logic works before touching the real dataset.
    """
    print(f"\n{'═'*60}")
    print("  SANITY CHECKS — hand-crafted sensor values")
    print(f"{'═'*60}\n")

    def make_reading(**kwargs):
        defaults = dict(
            product_id="TEST-001", machine_type="M",
            air_temp_k=298.0, process_temp_k=308.0,
            rpm=1500.0, torque_nm=40.0, tool_wear_min=0.0,
            actual_failure=0, actual_twf=0, actual_hdf=0,
            actual_pwf=0, actual_osf=0, actual_rnf=0,
        )
        defaults.update(kwargs)
        return SensorReading(**defaults)

    tests = [
        # (description, reading, expected_state, expected_type)
        (
            "Normal operation",
            make_reading(),
            "NORMAL", None
        ),
        (
            "PWF — power too high (torque=70, rpm=1300 → ~9534W)",
            make_reading(torque_nm=70.0, rpm=1300.0),
            "CRITICAL", "PWF"
        ),
        (
            "PWF — power too low (torque=3, rpm=1000 → ~314W)",
            make_reading(torque_nm=3.0, rpm=1000.0),
            "CRITICAL", "PWF"
        ),
        (
            "TWF — tool wear in critical zone (225 min)",
            make_reading(tool_wear_min=225.0),
            "CRITICAL", "TWF"
        ),
        (
            "TWF — tool wear in warning zone (205 min)",
            make_reading(tool_wear_min=205.0),
            "WARNING", "TWF"
        ),
        (
            "OSF — Type L overstrain (wear=190, torque=58 → 11020, power safe, no TWF)",
            make_reading(machine_type="L", tool_wear_min=190.0, torque_nm=58.0, rpm=1000.0),
            "CRITICAL", "OSF"
        ),
        (
            "HDF — cooling failure (delta=8K, rpm=1300)",
            make_reading(air_temp_k=302.0, process_temp_k=310.0, rpm=1300.0),
            "CRITICAL", "HDF"
        ),
        (
            "HDF — only ONE condition (low delta but normal RPM) → NORMAL",
            make_reading(air_temp_k=302.0, process_temp_k=310.0, rpm=1600.0),
            "NORMAL", None
        ),
        (
            "Priority — PWF + TWF both fire → PWF wins (priority 1 > 2)",
            make_reading(torque_nm=70.0, rpm=1300.0, tool_wear_min=225.0),
            "CRITICAL", "PWF"
        ),
    ]

    passed = 0
    for desc, reading, exp_state, exp_type in tests:
        output = run_detectors(reading, timestamp=0)
        state_ok = output.overall_state == exp_state
        type_ok  = (exp_type is None and output.primary_alert is None) or \
                   (output.primary_alert is not None and
                    output.primary_alert.failure_type == exp_type)
        ok = state_ok and type_ok
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {desc}")
        if not ok:
            print(f"       Expected: state={exp_state}, type={exp_type}")
            print(f"       Got:      state={output.overall_state}, "
                  f"type={output.primary_alert.failure_type if output.primary_alert else None}")
            if output.primary_alert:
                print(f"       Message: {output.primary_alert.message}")
        elif output.primary_alert:
            print(f"       → {output.primary_alert.message[:80]}...")
        passed += int(ok)

    print(f"\n  {passed}/{len(tests)} sanity checks passed\n")
    return passed == len(tests)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the anomaly detection engine")
    parser.add_argument("--path",   default="../data/ai4i2020.csv", help="Path to CSV")
    parser.add_argument("--rows",   type=int, default=None,         help="Limit rows tested")
    parser.add_argument("--quiet",  action="store_true",            help="Hide per-row output")
    parser.add_argument("--sanity", action="store_true",            help="Run sanity checks only")
    args = parser.parse_args()

    # Always run sanity checks first
    all_passed = run_sanity_checks()

    if not args.sanity and all_passed:
        run_tests(
            csv_path = args.path,
            max_rows = args.rows,
            verbose  = not args.quiet,
        )
    elif not all_passed:
        print("⛔  Sanity checks failed — fix detector.py before running on real data.\n")
        sys.exit(1)
