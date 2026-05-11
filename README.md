# 🏭 Factory Floor Anomaly Detector

Containerized IoT pipeline that detects CNC machine failures before they occur using physics-based rules on real sensor data.

**Stack:** Python · Docker · MQTT · InfluxDB · Telegraf · Grafana  
**Dataset:** [AI4I 2020 Predictive Maintenance](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

---

## 🏗️ Pipeline

```
CSV → Publisher → Mosquitto → Telegraf → InfluxDB → Grafana
```

---

## 🔬 What It Detects

4 physics-based detectors, each with NORMAL → WARNING → CRITICAL states:

| Detector | Trigger | Priority |
|---|---|---|
| PWF — Power Failure | torque × rpm outside 3500–9000W | 1 (highest) |
| TWF — Tool Wear | tool wear > 200 min | 2 |
| OSF — Overstrain | tool_wear × torque > type limit | 3 |
| HDF — Heat Dissipation | temp delta < 8.6K AND rpm < 1380 | 4 |

When multiple fire simultaneously → most dangerous wins.

---

## 🚀 Run

```bash
# Add dataset first
# Download ai4i2020.csv → place in data/

docker-compose up --build
```

| Service | URL | Login |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin123 |
| InfluxDB | http://localhost:8086 | admin / adminpassword123 |

```bash
# Stop
docker-compose down

# Stop + wipe data
docker-compose down -v

# Debug MQTT messages
docker exec -it mosquitto mosquitto_sub -t "factory/#" -v

# Test detector logic only (no Docker needed)
python publisher/test_detector.py --sanity
```

---

## 📁 Structure

```
├── docker-compose.yml
├── data/                    ← place ai4i2020.csv here
├── publisher/
│   ├── detector.py          ← detection logic
│   ├── publisher.py         ← MQTT publisher
│   └── test_detector.py     ← tests
├── mosquitto/config/
├── telegraf/
└── grafana/provisioning/
```
