import os
import json
from fastapi import APIRouter, HTTPException
from config import OUTPUTS_DIR, DATA_DIR, PRIMARY_DATASET, SECONDARY_DATASET, MODELS_DIR

router = APIRouter(prefix="/api", tags=["Data"])


@router.get("/datasets/info")
def get_datasets_info():
    """Return info about loaded datasets."""
    summary_path = os.path.join(OUTPUTS_DIR, "eda_summary.json")

    datasets = []

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

        for key, ds in summary.items():
            datasets.append(
                {
                    "name": ds.get("name", key),
                    "filename": ds.get("filename", ""),
                    "records": ds.get("total_records", 0),
                    "features": ds.get("total_features", 0),
                    "severity_classes": ds.get("severity_classes", 0),
                    "class_distribution": ds.get("class_distribution", {}),
                    "columns": ds.get("columns", []),
                    "status": "loaded",
                }
            )
    else:
        primary_exists = os.path.exists(os.path.join(DATA_DIR, PRIMARY_DATASET))
        secondary_exists = os.path.exists(os.path.join(DATA_DIR, SECONDARY_DATASET))

        datasets.append(
            {
                "name": "NHAI Multi-Corridor (ETP_4_New_Data_Accidents)",
                "filename": PRIMARY_DATASET,
                "records": 0,
                "features": 0,
                "severity_classes": 0,
                "status": "found" if primary_exists else "not_found",
                "download_url": (
                    "https://doi.org/10.5281/zenodo.16946653"
                    if not primary_exists
                    else None
                ),
            }
        )

        datasets.append(
            {
                "name": "Kaggle India Severity (Road)",
                "filename": SECONDARY_DATASET,
                "records": 0,
                "features": 0,
                "severity_classes": 0,
                "status": "found" if secondary_exists else "not_found",
                "download_url": (
                    "https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india"
                    if not secondary_exists
                    else None
                ),
            }
        )

    return {"datasets": datasets}


@router.get("/filters/options")
def get_filter_options():
    """Return all unique values for filter dropdowns."""
    summary_path = os.path.join(OUTPUTS_DIR, "eda_summary.json")

    default_options = {
        "Day_of_Week": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        "Weather_Conditions": ["Clear", "Rainy", "Foggy", "Cloudy", "Windy", "Other"],
        "Vehicle_Types": [
            "Car",
            "Truck",
            "Bus",
            "Two Wheeler",
            "Auto Rickshaw",
            "Other",
        ],
        "Road_Conditions": ["Straight", "Curve", "Bridge", "Intersection", "Other"],
        "Causes": [
            "Overspeeding",
            "Drunk Driving",
            "Wrong Side Driving",
            "Distracted Driving",
            "Red Light Jumping",
            "Tire Burst",
            "Poor Visibility",
            "Road Defect",
            "Other",
        ],
        "Time_Periods": ["Morning", "Afternoon", "Evening", "Night"],
        "Nature_of_Accident": [
            "Head-on Collision",
            "Rear-end Collision",
            "Side Collision",
            "Hit and Run",
            "Overturning",
            "Pedestrian Knock Down",
            "Other",
        ],
        "Intersection_Types": [
            "None",
            "T-Junction",
            "Y-Junction",
            "Four-way",
            "Roundabout",
            "Other",
        ],
        "Accident_Location": ["Urban", "Rural"],
        "Models": [
            "RandomForest",
            "XGBoost",
            "GradientBoosting",
            "SVM",
            "LogisticRegression",
        ],
    }

    filter_data_path = os.path.join(OUTPUTS_DIR, "filter_options.json")
    if os.path.exists(filter_data_path):
        with open(filter_data_path) as f:
            saved_options = json.load(f)
        default_options.update(saved_options)

    return default_options


@router.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns status of models, digital twins and datasets.
    """
    import datetime
    from config import CITIES_CONFIG

    # Check models
    models_exist = (
        os.path.exists(MODELS_DIR)
        and any(f.endswith(".joblib") for f in os.listdir(MODELS_DIR))
        if os.path.exists(MODELS_DIR)
        else False
    )

    # Check digital twin status
    twin_status = {}
    try:
        from api.routes_digital_twin import digital_twins

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
    except Exception:
        twin_status = {}

    return {
        "status": "healthy",
        "models_loaded": models_exist,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "datasets": {
            "primary": os.path.exists(os.path.join(DATA_DIR, PRIMARY_DATASET)),
            "secondary": os.path.exists(os.path.join(DATA_DIR, SECONDARY_DATASET)),
        },
        "digital_twins": twin_status,
        "version": "2.0.0",
    }


@router.get("/data/preview/{dataset_key}")
def get_data_preview(dataset_key: str = "primary", page: int = 1, per_page: int = 25):
    """Return paginated raw data preview."""
    import pandas as pd

    if dataset_key == "primary":
        filepath = os.path.join(DATA_DIR, PRIMARY_DATASET)
    elif dataset_key == "secondary":
        filepath = os.path.join(DATA_DIR, SECONDARY_DATASET)
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid dataset_key: {dataset_key}"
        )

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404, detail=f"Dataset file not found: {filepath}"
        )

    try:
        df = pd.read_csv(filepath)
        total_records = len(df)
        total_pages = (total_records + per_page - 1) // per_page

        start = (page - 1) * per_page
        end = start + per_page
        page_data = df.iloc[start:end]

        records = page_data.fillna("").to_dict(orient="records")

        return {
            "columns": list(df.columns),
            "records": records,
            "total_records": total_records,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
