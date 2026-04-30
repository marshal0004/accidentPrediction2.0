# Delhi Accident Severity Predictor — Local Setup Guide

Complete guide for running the Delhi Accident Prediction project locally in VS Code.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.10+ | `python3 --version` |
| Node.js | 16+ | `node --version` |
| VS Code | Latest | `code --version` |
| Git | 2.0+ | `git --version` |

### Recommended VS Code Extensions

- **Python** — IntelliSense, linting, debugging
- **ESLint** — JavaScript linting
- **Prettier** — Code formatting
- **Thunder Client** or **REST Client** — API testing

---

## 2. Clone and Install

```bash
git clone https://github.com/marshal0004/accidentPrediction
cd accidentPrediction/accident-severity-predictor
```

### Project Structure

```
accident-severity-predictor/
├── backend/           # FastAPI Python backend
│   ├── main.py        # FastAPI app
│   ├── config.py      # Configuration
│   ├── ml/            # ML modules
│   │   ├── delhi_data_mapper.py  # Loads ALL Delhi datasets
│   │   ├── delhi_trainer.py      # Trains ML models
│   │   ├── digital_twin.py       # Digital twin orchestrator
│   │   ├── segment_risk_calculator.py
│   │   ├── road_network_loader.py
│   │   ├── predictor.py
│   │   └── ...
│   ├── api/           # API routes
│   ├── data/          # Data files
│   │   ├── delhiDatasets/  # 10 real Delhi accident datasets
│   │   ├── road_networks/  # OSM road network
│   │   └── mapped_accidents/
│   ├── outputs/       # Model outputs, charts
│   ├── tests/         # Test files
│   └── requirements.txt
├── frontend/          # React frontend
│   ├── src/
│   ├── package.json
│   └── ...
└── localsetupfinal.md  # THIS FILE
```

---

## 3. Backend Setup

### 3.1 Create Virtual Environment

```bash
cd backend
python -m venv venv
```

### 3.2 Activate Virtual Environment

```bash
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3.3 Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key Python packages installed:**

| Package | Purpose |
|---|---|
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `scikit-learn` | ML models (Random Forest, SVM, Logistic Regression) |
| `xgboost` | Gradient boosting model |
| `imbalanced-learn` | SMOTE for class imbalance |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `geopandas` | Geospatial data handling |
| `osmnx` | OpenStreetMap network download |
| `joblib` | Model serialization |
| `shap` | Model explainability |
| `openpyxl` | Excel file reading |
| `folium` | Map visualization |

### 3.4 Verify Installation

```bash
python -c "import fastapi; import sklearn; import xgboost; import osmnx; print('All imports OK')"
```

> **Note:** If `osmnx` or `geopandas` installation fails, you may need system-level
> GEOS and GDAL libraries. On Ubuntu: `sudo apt install libgeos-dev libgdal-dev`.

---

## 4. Frontend Setup

```bash
cd ../frontend
npm install
```

> If `npm install` fails with dependency errors:
> ```bash
> rm -rf node_modules package-lock.json
> npm cache clean --force
> npm install --legacy-peer-deps
> ```

---

## 5. Step-by-Step Run Order (IMPORTANT)

> **These are long-running tasks.** Execute them in the exact order below.
> Steps 1–4 are **first-time only** setup. Steps 5–6 are for starting the servers each session.

### Step 1: Download Delhi Road Network (5–10 min)

First time only — downloads the OSM road network for all of Delhi NCT.

```bash
cd backend
source venv/bin/activate
python -c "from ml.road_network_loader import RoadNetworkLoader; loader = RoadNetworkLoader('delhi'); loader.get_or_download_network()"
```

> **IMPORTANT:** Delete the old cached network first if it exists:
> ```bash
> rm -rf data/road_networks/delhi/
> ```
> This ensures the **FULL Delhi NCT road network** is downloaded (not just Central Delhi).

### Step 2: Map Real Delhi Accident Data (2–5 min)

Maps ALL 10 Delhi datasets to road segments:

```bash
python -c "from ml.delhi_data_mapper import DelhiDataMapper; from ml.road_network_loader import RoadNetworkLoader; loader = RoadNetworkLoader('delhi'); edges = loader.get_edges_gdf(); mapper = DelhiDataMapper(edges, 'delhi'); mapper.geocode_and_map_all()"
```

This loads:

| Dataset | Description |
|---|---|
| Dataset 1 | 77 Delhi rows with real GPS coordinates |
| Dataset 2 | 2,433 Delhi rows with real GPS coordinates |
| Dataset 3 | ALL 30 CSV files (2021–2023 crash data) |
| Dataset 4 | 2019–2021 ML-ready data |
| Dataset 5 | ALL 8 CSV files (2016 data) |
| Dataset 6 | Circle-wise data |
| Dataset 7 | ALL 21 CSV files (2018 data) |
| Dataset 8 | 2022–2024 classification |
| Dataset 9 | 2020–2022 data |
| Dataset 10 | Comprehensive 2016–2020 data |

### Step 3: Build Digital Twin (1–3 min)

```bash
python -c "from ml.digital_twin import DigitalTwin; twin = DigitalTwin('delhi'); twin.build()"
```

### Step 4: Train ML Models (10–30 min)

Trains 5 models with hyperparameter tuning:

```bash
python -c "from ml.delhi_trainer import DelhiTrainer; trainer = DelhiTrainer(); results = trainer.run()"
```

**Models trained:**

| Model | Type |
|---|---|
| XGBoost | Gradient Boosting |
| GradientBoosting | Scikit-learn Ensemble |
| RandomForest | Bagging Ensemble |
| SVM | Support Vector Machine |
| LogisticRegression | Linear Classifier |

All models use **SMOTE balancing** and **hyperparameter tuning**.

### Step 5: Start Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for the **"Server running"** message. The API docs are available at http://localhost:8000/docs.

### Step 6: Start Frontend

```bash
cd frontend
npm start
```

Opens at **http://localhost:3000**

### Step 7: (Optional) Use Caddy Gateway for API Proxy

If the frontend can't reach the backend directly:

```bash
caddy run --config Caddyfile
```

This proxies `/api/*` to `localhost:8000` on port 81.

---

## 6. Running Tests

```bash
cd backend
source venv/bin/activate

# Unit tests
python -m pytest tests/test_unit.py -v

# Integration tests
python -m pytest tests/test_integration.py -v

# System tests
python -m pytest tests/test_system.py -v

# All tests
python -m pytest tests/ -v

# Run with markers
python -m pytest tests/ -v -m "not slow"      # Skip slow tests
python -m pytest tests/ -v -m functional       # Functional tests only
python -m pytest tests/ -v -m nonfunctional    # Non-functional tests only
python -m pytest tests/ -v -m performance      # Performance tests only
```

---

## 7. Color Scheme for Risk Map

| Risk Range | Color | Category |
|---|---|---|
| 0–10% | 🟢 Green (`#22C55E`) | Zero Accidents |
| 10–40% | 🔵 Blue (`#3B82F6`) | Low Risk |
| 40–60% | 🟡 Yellow (`#EAB308`) | Moderate Risk |
| 60–80% | 🟠 Orange (`#F97316`) | High Risk |
| 80–95%+ | 🔴 Red (`#EF4444`) | Very High Risk |

---

## 8. Data Overview

- **10 real Delhi accident datasets** (2016–2024)
- **~8,000+** total accident records
- Maps to **500+ road segments** across Delhi NCT
- Virtual segments created for GPS points outside the road network
- **5 ML models** trained with SMOTE balancing and hyperparameter tuning

---

## 9. Troubleshooting

### Road Network Download Fails

**Problem:** OSMnx times out when downloading the Delhi road network.

**Solution:** Check your internet connection and try again. OSM servers can be intermittent. The system automatically retries with backoff.

### Mapping Gives Low Counts

**Problem:** Fewer accident records mapped than expected.

**Solution:** Delete the cached mapping data and re-run Step 2:
```bash
rm -rf data/mapped_accidents/delhi/
```
Then re-run the mapping command from Step 2.

### ML Training Fails

**Problem:** Training script crashes with a file-not-found or data error.

**Solution:** Check that `segment_mapping.json` exists in `data/mapped_accidents/delhi/`. If not, re-run Step 2 first.

### Frontend Can't Reach Backend

**Problem:** Frontend on port 3000 cannot connect to backend on port 8000.

**Solution:**
- Use the Caddy gateway (Step 7)
- Or set a proxy in `frontend/package.json`:
  ```json
  { "proxy": "http://localhost:8000" }
  ```
- Or configure `frontend/src/api/apiClient.js`:
  ```javascript
  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  ```

### Models Not Found

**Problem:** `Model not found` or `No models available` when trying to predict.

**Solution:** Re-run Step 4 (training) to generate model artifacts.

### Port Already in Use (8000)

**Problem:** `Address already in use` error when starting backend.

**Solution:**
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Or use a different port:
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### npm Install Failures

**Problem:** `npm install` fails with dependency errors.

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install --legacy-peer-deps
```

### Geocoding Returns None

**Problem:** Many locations return `None, None` from geocode_location().

**Solution:** This is expected for locations not in the `DELHI_KNOWN_LOCATIONS` database. The system uses a hardcoded dictionary of ~150 Delhi locations. Unknown locations that can't be matched by partial/word matching will return None. Unmapped records are gracefully skipped during segment mapping.

### Memory Issues

**Problem:** Python process uses too much memory during twin initialization.

**Solution:** The Delhi road network has ~30,000+ segments. First-time build can use 2–4 GB RAM. Ensure your system has at least 4 GB free memory.

---

*Last updated: 2025*
