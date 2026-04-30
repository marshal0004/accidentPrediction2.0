# backend/main.py

import os
import sys
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    API_HOST,
    API_PORT,
    DATA_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    PRIMARY_DATASET,
    SECONDARY_DATASET,
    PRIMARY_PATH,
    SECONDARY_PATH,
    CITIES_CONFIG,
    DEFAULT_CITY,
)
from database import create_tables

# ─────────────────────────────────────────────
# EXISTING ROUTERS
# ─────────────────────────────────────────────
from api.routes_eda import router as eda_router
from api.routes_models import router as models_router
from api.routes_shap import router as shap_router
from api.routes_predict import router as predict_router
from api.routes_data import router as data_router

# ─────────────────────────────────────────────
# NEW DIGITAL TWIN ROUTERS
# ─────────────────────────────────────────────
from api.routes_digital_twin import router as digital_twin_router
from api.routes_digital_twin import digital_twins
from api.routes_what_if import router as what_if_router

# ─────────────────────────────────────────────
# GLOBAL PREDICTOR INSTANCE
# (Shared between existing routes and digital twin)
# ─────────────────────────────────────────────
predictor_instance = None

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Indian Road Accident Severity Prediction API",
    description=(
        "ML-powered API for predicting road accident severity in India. "
        "Includes Digital Twin of road network with risk heatmaps, "
        "scenario simulation and what-if analysis."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# REGISTER ROUTERS
# ─────────────────────────────────────────────

# Existing routers
app.include_router(eda_router)
app.include_router(models_router)
app.include_router(shap_router)
app.include_router(predict_router)
app.include_router(data_router)

# New digital twin routers
app.include_router(digital_twin_router)
app.include_router(what_if_router)


# ─────────────────────────────────────────────
# STARTUP EVENT
# ─────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """
    Run on server startup:
    1. Create DB tables
    2. Check datasets
    3. Load ML models
    4. Load digital twins (if cache exists)
    """
    global predictor_instance

    print("\n" + "=" * 60)
    print("  Indian Road Accident Severity Prediction API v2.0")
    print("  Starting up...")
    print("=" * 60)

    # ─────────────────────────────────
    # Step 1: Database
    # ─────────────────────────────────
    create_tables()
    print("  [OK] Database tables created/verified")

    # ─────────────────────────────────
    # Step 2: Check datasets
    # ─────────────────────────────────
    primary_exists = os.path.exists(PRIMARY_PATH)
    secondary_exists = os.path.exists(SECONDARY_PATH)

    print(f"\n  Dataset Status:")
    print(
        f"    Primary   ({PRIMARY_DATASET}): "
        f"{'✓ Found' if primary_exists else '✗ NOT FOUND'}"
    )
    print(
        f"    Secondary ({SECONDARY_DATASET}): "
        f"{'✓ Found' if secondary_exists else '✗ NOT FOUND'}"
    )

    if not primary_exists:
        print(f"\n  [DOWNLOAD] Primary dataset:")
        print(f"    URL: https://doi.org/10.5281/zenodo.16946653")
        print(f"    Save as: {PRIMARY_PATH}")

    if not secondary_exists:
        print(f"\n  [DOWNLOAD] Secondary dataset:")
        print(
            f"    URL: https://www.kaggle.com/datasets/"
            f"s3programmer/road-accident-severity-in-india"
        )
        print(f"    Save as: {SECONDARY_PATH}")

    # ─────────────────────────────────
    # Step 3: Load ML models
    # ─────────────────────────────────
    print(f"\n  Loading ML Models...")

    models_exist = (
        os.path.exists(MODELS_DIR)
        and any(f.endswith(".joblib") for f in os.listdir(MODELS_DIR))
        if os.path.exists(MODELS_DIR)
        else False
    )

    if models_exist:
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
        print(f"  [OK] {len(model_files)} model artifacts found:")
        for mf in model_files:
            print(f"    - {mf}")

        try:
            from api.routes_predict import get_predictor

            predictor = get_predictor()
            predictor_instance = predictor
            print(
                f"  [OK] Predictor loaded: "
                f"{len(predictor.get_available_models())} models ready"
            )
        except Exception as e:
            print(f"  [WARN] Predictor load failed: {e}")
            predictor_instance = None
    else:
        print(f"  [INFO] No trained models found in {MODELS_DIR}")
        print(f"  [INFO] Run 'python run_pipeline.py' to train models")
        print(f"  [INFO] Or POST to /api/train to trigger training")
        predictor_instance = None

    # ─────────────────────────────────
    # Step 4: Load Digital Twins
    # ─────────────────────────────────
    print(f"\n  Loading Digital Twins...")

    twin_status = {}

    for city_key in CITIES_CONFIG.keys():
        try:
            from ml.digital_twin import DigitalTwin

            twin = DigitalTwin(city_key, predictor=predictor_instance)

            # Try to load from cache (fast)
            # Don't build here - let user trigger via API
            network_cache_exists = _check_twin_cache(city_key)

            if network_cache_exists:
                print(f"  [OK] Twin cache found: {city_key} (lazy load on first request)")
                # Don't load twin data here - it's 14MB+ of JSON.
                # Instead, load lazily on first API request.
                digital_twins[city_key] = twin
                twin_status[city_key] = "ready"
                print(
                    f"  [OK] {city_key} twin available (lazy load)"
                )
            else:
                twin_status[city_key] = "not_initialized"
                print(
                    f"  [INFO] {city_key} twin not built yet. "
                    f"Call GET /api/twin/{city_key}/initialize to build."
                )

        except Exception as e:
            twin_status[city_key] = "error"
            print(f"  [WARN] {city_key} twin load failed: {e}")

    # ─────────────────────────────────
    # Step 5: Print all API endpoints
    # ─────────────────────────────────
    print(f"\n  Existing API Endpoints:")
    print(f"    GET  /api/health")
    print(f"    GET  /api/eda/summary")
    print(f"    GET  /api/eda/charts/{{name}}")
    print(f"    GET  /api/models/comparison")
    print(f"    GET  /api/shap/feature-importance")
    print(f"    POST /api/predict")
    print(f"    POST /api/predict/batch")
    print(f"    GET  /api/datasets/info")
    print(f"    GET  /api/data/preview/{{key}}")

    print(f"\n  Digital Twin API Endpoints:")
    print(f"    GET  /api/twin/cities")
    print(f"    GET  /api/twin/{{city}}/initialize")
    print(f"    GET  /api/twin/{{city}}/metadata")
    print(f"    GET  /api/twin/{{city}}/heatmap")
    print(f"    GET  /api/twin/{{city}}/segments/top-dangerous")
    print(f"    GET  /api/twin/{{city}}/segment/{{id}}")
    print(f"    POST /api/twin/{{city}}/segment/{{id}}/simulate")
    print(f"    POST /api/twin/{{city}}/refresh")
    print(f"    GET  /api/twin/{{city}}/stats")

    print(f"\n  What-If API Endpoints:")
    print(f"    GET  /api/what-if/interventions")
    print(f"    POST /api/what-if/{{city}}/segment/{{id}}/analyze")
    print(f"    POST /api/what-if/{{city}}/segment/{{id}}/compare")
    print(f"    POST /api/what-if/{{city}}/batch-analyze")
    print(f"    GET  /api/what-if/{{city}}/recommendations")

    print(f"\n  Digital Twin Status:")
    for city, status in twin_status.items():
        icon = "✓" if status == "ready" else "○"
        print(f"    {icon} {city}: {status}")

    print(f"\n  Server running at http://{API_HOST}:{API_PORT}")
    print(f"  Docs at http://localhost:{API_PORT}/docs")
    print("=" * 60 + "\n")


def _check_twin_cache(city_key: str) -> bool:
    """
    Check if digital twin cache exists for a city.

    Args:
        city_key: City identifier

    Returns:
        True if cache exists and is valid
    """
    try:
        from ml.road_network_loader import RoadNetworkLoader
        from ml.accident_segment_mapper import AccidentSegmentMapper
        from ml.segment_risk_calculator import SegmentRiskCalculator

        # Check road network cache
        loader = RoadNetworkLoader(city_key)
        if not loader.is_cache_valid():
            return False

        # Check mapped accidents cache
        import geopandas as gpd
        from config import MAPPED_ACCIDENTS_DIR

        mapping_path = os.path.join(
            MAPPED_ACCIDENTS_DIR, city_key, "segment_mapping.json"
        )
        if not os.path.exists(mapping_path):
            return False

        # Check risk scores cache
        from config import DIGITAL_TWIN_DIR

        risks_path = os.path.join(DIGITAL_TWIN_DIR, city_key, "segment_risks.json")
        if not os.path.exists(risks_path):
            return False

        return True

    except Exception as e:
        return False


# ─────────────────────────────────────────────
# STATIC FRONTEND (React Build)
# ─────────────────────────────────────────────

FRONTEND_BUILD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "frontend", "build"
)


# ─────────────────────────────────────────────
# ROOT ENDPOINT – serves React app if available
# ─────────────────────────────────────────────


@app.get("/")
def root():
    """Root endpoint – serves React app if built, else API info."""
    index_html = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {
        "name": "Indian Road Accident Severity Prediction API",
        "version": "2.0.0",
        "docs": f"http://localhost:{API_PORT}/docs",
        "health": f"http://localhost:{API_PORT}/api/health",
        "features": [
            "Accident Severity Prediction",
            "Digital Twin of Road Network",
            "Risk Heatmaps",
            "Scenario Simulation",
            "What-If Analysis",
        ],
    }


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────


@app.get("/api/health")
def health_check():
    """
    Health check endpoint.

    Returns status of:
    - ML models
    - Digital twins
    - Datasets
    """
    # Check models
    models_loaded = predictor_instance is not None
    available_models = []

    if models_loaded:
        try:
            available_models = predictor_instance.get_available_models()
        except Exception:
            pass

    # Check digital twins
    twin_status = {}
    for city_key in CITIES_CONFIG.keys():
        if city_key in digital_twins:
            twin = digital_twins[city_key]
            meta = twin.get_metadata()
            twin_status[city_key] = {
                "status": meta.get("status", "unknown"),
                "total_segments": meta.get("total_segments", 0),
                "high_risk_segments": meta.get("high_risk_segments", 0),
            }
        else:
            twin_status[city_key] = {
                "status": "not_initialized",
                "total_segments": 0,
                "high_risk_segments": 0,
            }

    # Check datasets
    datasets = {
        "primary": os.path.exists(PRIMARY_PATH),
        "secondary": os.path.exists(SECONDARY_PATH),
    }

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "models_loaded": models_loaded,
        "available_models": available_models,
        "digital_twins": twin_status,
        "datasets": datasets,
    }


# ─────────────────────────────────────────────
# TRAIN ENDPOINT
# ─────────────────────────────────────────────


@app.post("/api/train")
def trigger_training():
    """Trigger the full ML training pipeline."""
    global predictor_instance

    try:
        from ml.data_loader import load_all_datasets, generate_eda_summary
        from ml.preprocessor import preprocess_dataset
        from ml.trainer import train_all_models
        from ml.evaluator import evaluate_all_models, generate_evaluation_plots
        from ml.shap_analyzer import run_shap_all_models

        datasets = load_all_datasets()
        if not datasets:
            raise HTTPException(status_code=404, detail="No datasets found in ./data/")

        generate_eda_summary(datasets)

        all_results = {}
        for key, ds in datasets.items():
            df = ds["dataframe"]
            target = ds["target_column"]
            roles = ds["roles"]

            X, y, feature_names, le, label_mapping = preprocess_dataset(
                df, target, roles, key
            )

            results, X_train, X_test, y_train, y_test, X_smote, y_smote = (
                train_all_models(X, y, feature_names, key)
            )

            all_metrics = evaluate_all_models(
                results, X_test, y_test, label_mapping, key
            )
            generate_evaluation_plots(
                all_metrics, results, X_test, y_test, label_mapping, key
            )

            run_shap_all_models(results, X_test, feature_names, label_mapping, key)

            all_results[key] = {
                "models_trained": len(
                    [r for r in results.values() if r.get("model") is not None]
                ),
                "best_model": (
                    max(all_metrics, key=lambda x: x["f1_weighted"])["model_name"]
                    if all_metrics
                    else None
                ),
            }

        # Reload predictor after training
        from api.routes_predict import get_predictor

        predictor = get_predictor()
        predictor.load_artifacts()
        predictor_instance = predictor

        # Update digital twins with new predictor
        for city_key, twin in digital_twins.items():
            twin.predictor = predictor_instance
            if twin.scenario_simulator:
                twin.scenario_simulator.predictor = predictor_instance
            print(f"  [OK] Updated predictor for {city_key} digital twin")

        return {
            "status": "Training complete",
            "results": all_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ─────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ─────────────────────────────────────────────
# MOUNT STATIC FRONTEND FILES
# ─────────────────────────────────────────────
if os.path.isdir(FRONTEND_BUILD_DIR):
    # Serve /static/ from React build
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "static")),
        name="react-static",
    )
    # Serve other React assets (manifest, favicon, logos, etc.)
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "static")),
        name="react-assets",
    )

    # Catch-all for React Router – serve index.html for any non-API route
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all non-API routes (client-side routing)."""
        # Don't intercept API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        # Check if a specific static file exists
        file_path = os.path.join(FRONTEND_BUILD_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise serve index.html (for React Router)
        index_html = os.path.join(FRONTEND_BUILD_DIR, "index.html")
        if os.path.exists(index_html):
            return FileResponse(index_html)
        raise HTTPException(status_code=404, detail="Not found")

    print(f"  [OK] React frontend served from {FRONTEND_BUILD_DIR}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False, log_level="info")
