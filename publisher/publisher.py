"""
publisher.py — Factory Floor Anomaly Detector
==============================================
Role in the pipeline:
    CSV file → [this file] → MQTT broker → Telegraf → InfluxDB → Grafana

This file does exactly three things:
    1. Reads the AI4I dataset row by row
    2. Runs each row through the detection engine (detector.py)
    3. Publishes results to MQTT topics

MQTT Topics:
    factory/sensor/raw      → every row, always
    factory/alert/primary   → only WARNING or CRITICAL
"""

import os
import sys
import time
import json
import pandas as pd
import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import SensorReading, run_detectors

# ── Config ────────────────────────────────────────────────────────────────────
BROKER_HOST   = os.getenv("MQTT_BROKER",   "mosquitto")
BROKER_PORT   = int(os.getenv("MQTT_PORT", "1883"))
DATA_PATH     = os.getenv("DATA_PATH",    r"C:\Navya\Project 2026\Factory anamoly detection\data\ai4i2020.csv")
REPLAY_DELAY  = float(os.getenv("REPLAY_DELAY", "1.0"))

TOPIC_SENSORS = "factory/sensor/raw"
TOPIC_ALERTS  = "factory/alert/primary"

# ── MQTT Callbacks ────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[publisher] Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
    else:
        print(f"[publisher] Connection refused — code {rc}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"[publisher] Unexpected disconnect (rc={rc}). Will retry...")

def on_publish(client, userdata, mid):
    pass

# ── Retry Connection ──────────────────────────────────────────────────────────
def connect_with_retry(client, host, port, max_attempts=10):
    attempt = 0
    while attempt < max_attempts:
        try:
            print(f"[publisher] Connecting to {host}:{port} (attempt {attempt+1}/{max_attempts})...")
            client.connect(host, port, keepalive=60)
            client.loop_start()
            time.sleep(1)
            return True
        except Exception as e:
            attempt += 1
            wait = 2 * attempt
            print(f"[publisher] Failed — {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return False

# ── Row Parser ────────────────────────────────────────────────────────────────
def parse_row(row) -> SensorReading:
    return SensorReading(
        product_id     = str(row.get("product id",   row.get("product_id", "unknown"))),
        machine_type   = str(row.get("type", "M")).strip().upper(),
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

# ── Payload Builders ──────────────────────────────────────────────────────────
def build_sensor_payload(reading, output, timestamp):
    return {
        "timestamp":      timestamp,
        "product_id":     reading.product_id,
        "machine_type":   reading.machine_type,
        "air_temp_k":     reading.air_temp_k,
        "process_temp_k": reading.process_temp_k,
        "rpm":            reading.rpm,
        "torque_nm":      reading.torque_nm,
        "tool_wear_min":  reading.tool_wear_min,
        "overall_state":  output.overall_state,
    }

def build_alert_payload(output, timestamp):
    p = output.primary_alert
    return {
        "timestamp":       timestamp,
        "product_id":      output.product_id,
        "machine_type":    output.machine_type,
        "failure_type":    p.failure_type,
        "state":           p.state,
        "message":         p.message,
        "sensor_values":   p.sensor_values,
        "secondary_risks": output.secondary_risks,
        "actual_failure":  output.actual_failure,
        "early_warning":   output.early_warning,
    }

# ── Logger ────────────────────────────────────────────────────────────────────
STATE_ICON = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}

def log_row(row_count, reading, output):
    is_alert = output.overall_state != "NORMAL"
    if is_alert or row_count % 50 == 0:
        icon = STATE_ICON.get(output.overall_state, "?")
        line = (f"[row {row_count:5d}] {icon} {output.overall_state:8s} | "
                f"type={reading.machine_type}  rpm={reading.rpm:6.0f}  "
                f"torque={reading.torque_nm:5.1f}Nm  wear={reading.tool_wear_min:4.0f}min")
        if is_alert and output.primary_alert:
            p = output.primary_alert
            line += f"\n         [{p.failure_type}] {p.message[:70]}..."
            if output.early_warning:
                line += f"\n         EARLY WARNING — actual_failure still 0"
        print(line)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n[publisher] Factory Floor Anomaly Detector — Publisher Starting")
    print(f"[publisher] Broker : {BROKER_HOST}:{BROKER_PORT}")
    print(f"[publisher] Data   : {DATA_PATH}")
    print(f"[publisher] Delay  : {REPLAY_DELAY}s per row\n")

    client = mqtt.Client(client_id="factory-publisher")
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish    = on_publish

    if not connect_with_retry(client, BROKER_HOST, BROKER_PORT):
        print("[publisher] Could not connect. Exiting.")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"[publisher] Dataset not found at {DATA_PATH}")
        sys.exit(1)

    df.columns = df.columns.str.strip().str.lower()
    print(f"[publisher] Loaded {len(df):,} rows. Streaming started...\n")

    row_count = 0
    while True:
        for _, row in df.iterrows():
            row_count += 1
            timestamp = int(time.time())

            reading = parse_row(row)
            output  = run_detectors(reading, timestamp)

            # Always publish raw sensor data
            client.publish(TOPIC_SENSORS, json.dumps(build_sensor_payload(reading, output, timestamp)), qos=1)

            # Only publish alert when something is wrong
            if output.overall_state != "NORMAL":
                client.publish(TOPIC_ALERTS, json.dumps(build_alert_payload(output, timestamp)), qos=1)

            log_row(row_count, reading, output)
            time.sleep(REPLAY_DELAY)

        print(f"\n[publisher] Dataset complete ({row_count} rows). Restarting...\n")

if __name__ == "__main__":
    main()
