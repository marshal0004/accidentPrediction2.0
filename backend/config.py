# backend/config.py
import os

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
SHAP_DIR = os.path.join(OUTPUTS_DIR, "shap")

# ─────────────────────────────────────────────
# DIGITAL TWIN PATHS
# ─────────────────────────────────────────────
ROAD_NETWORKS_DIR = os.path.join(DATA_DIR, "road_networks")
MAPPED_ACCIDENTS_DIR = os.path.join(DATA_DIR, "mapped_accidents")
DIGITAL_TWIN_DIR = os.path.join(OUTPUTS_DIR, "digital_twin")
GEOCODE_CACHE_DIR = os.path.join(DATA_DIR, "geocode_cache")

# ─────────────────────────────────────────────
# DELHI DATASETS PATH
# ─────────────────────────────────────────────
DELHI_DATASETS_DIR = os.path.join(DATA_DIR, "delhiDatasets")

# ─────────────────────────────────────────────
# DATASET PATHS (original - kept for backward compat)
# ─────────────────────────────────────────────
PRIMARY_DATASET = "ETP_4_New_Data_Accidents.csv"
SECONDARY_DATASET = "Road.csv"

PRIMARY_PATH = os.path.join(DATA_DIR, PRIMARY_DATASET)
SECONDARY_PATH = os.path.join(DATA_DIR, SECONDARY_DATASET)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "app.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ─────────────────────────────────────────────
# ML SETTINGS
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODEL_NAMES = [
    "RandomForest",
    "XGBoost",
    "GradientBoosting",
    "SVM",
    "LogisticRegression",
]

SEVERITY_COLORS = {
    "Fatal": "#EF4444",
    "Grievous": "#F97316",
    "Minor": "#F59E0B",
    "No Injury": "#10B981",
}

# ─────────────────────────────────────────────
# API SETTINGS
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─────────────────────────────────────────────
# DIGITAL TWIN SETTINGS
# ─────────────────────────────────────────────
OSM_CACHE_EXPIRY_DAYS = 30
HEATMAP_GRID_SIZE = 50
MIN_ACCIDENTS_PER_SEGMENT = 1
MAX_SNAP_DISTANCE_METERS = 5000  # 5km for Delhi city mapping (covers all Delhi NCT)
DEFAULT_CITY = "delhi"

# ─────────────────────────────────────────────
# NH CORRIDOR MAPPING
# Location code 1 = NH-44 (Delhi to Chennai)
# Location code 2 = NH-48 (Delhi to Mumbai)
# ─────────────────────────────────────────────
NH_CORRIDORS = {
    1: {
        "nh_number": "NH-44",
        "name": "Delhi-Chennai Highway",
        "from": "Delhi",
        "to": "Chennai",
        "start_lat": 28.6139,
        "start_lon": 77.2090,
        "end_lat": 13.0827,
        "end_lon": 80.2707,
        "total_length_km": 3745,
        "direction_lat": -0.004134,
        "direction_lon": 0.000830,
    },
    2: {
        "nh_number": "NH-48",
        "name": "Delhi-Mumbai Highway",
        "from": "Delhi",
        "to": "Mumbai",
        "start_lat": 28.6139,
        "start_lon": 77.2090,
        "end_lat": 19.0760,
        "end_lon": 72.8777,
        "total_length_km": 1428,
        "direction_lat": -0.006700,
        "direction_lon": -0.003020,
    },
}

# ─────────────────────────────────────────────
# ETP_4 CODE MAPPINGS
# ─────────────────────────────────────────────

# Accident_Severity_C codes
SEVERITY_CODE_MAP = {1: "Fatal", 2: "Grievous", 3: "Minor", 4: "No Injury"}

# Weather_Conditions_H codes
WEATHER_CODE_MAP = {
    1: "Clear",
    2: "Mist/Fog",
    3: "Rain",
    4: "Snow",
    5: "Hail",
    6: "Dust Storm",
    7: "Strong Wind",
    8: "Cloudy",
    9: "Very Hot",
    10: "Very Cold",
    11: "Smoke/Smog",
    12: "Other",
}

# Road_Condition_F codes
ROAD_CONDITION_CODE_MAP = {
    1: "Dry",
    2: "Wet",
    3: "Snow/Ice",
    4: "Muddy",
    5: "Oil Spill",
    6: "Loose Gravel",
    7: "Waterlogged",
    8: "Other",
}

# Road_Feature_E codes
ROAD_FEATURE_CODE_MAP = {
    1: "Straight Road",
    2: "Curve",
    3: "Bridge",
    4: "Intersection",
    5: "Roundabout",
    6: "Speed Breaker",
    7: "Toll Plaza",
    8: "Other",
}

# Causes_D codes
CAUSES_CODE_MAP = {
    1: "Speeding",
    2: "Wrong Side Driving",
    3: "Drunk Driving",
    4: "Distracted Driving",
    5: "Poor Visibility",
    6: "Road Defect",
    7: "Vehicle Defect",
    8: "Other",
}

# Vehicle_Type codes
VEHICLE_CODE_MAP = {
    1: "Car/Jeep/Van",
    2: "Truck/Lorry",
    3: "Bus",
    4: "Two Wheeler",
    5: "Three Wheeler",
    6: "Tractor",
    7: "Pedestrian",
    8: "Cyclist",
    9: "Animal Cart",
    10: "Other",
}

# RoadSide codes
ROADSIDE_CODE_MAP = {1: "Left", 2: "Right"}

# ─────────────────────────────────────────────
# CITIES CONFIG FOR DIGITAL TWIN
# ─────────────────────────────────────────────
CITIES_CONFIG = {
    "delhi": {
        "name": "Delhi",
        "display_name": "Delhi, India",
        "osm_query": "Delhi, India",
        "bbox": {
            "lat_min": 28.40,
            "lat_max": 28.90,
            "lon_min": 76.80,
            "lon_max": 77.35,
        },
        "center": [28.6139, 77.2090],
        "zoom_level": 11,
        "network_type": "drive",
        "nh_corridors": [1, 2],
    },
    "dehradun": {
        "name": "Dehradun",
        "display_name": "Dehradun, India",
        "osm_query": "Dehradun, Uttarakhand, India",
        "bbox": {
            "lat_min": 30.2500,
            "lat_max": 30.4200,
            "lon_min": 77.9200,
            "lon_max": 78.1200,
        },
        "center": [30.3165, 78.0322],
        "zoom_level": 12,
        "network_type": "drive",
        "nh_corridors": [1],
    },
    "bangalore": {
        "name": "Bangalore",
        "display_name": "Bangalore, India",
        "osm_query": "Bangalore, Karnataka, India",
        "bbox": {
            "lat_min": 12.8340,
            "lat_max": 13.1440,
            "lon_min": 77.4600,
            "lon_max": 77.7800,
        },
        "center": [12.9716, 77.5946],
        "zoom_level": 11,
        "network_type": "drive",
        "nh_corridors": [],
    },
}

# ─────────────────────────────────────────────
# RISK CALCULATION SETTINGS
# Updated per user requirements:
# 0-10% = Green (zero accidents)
# 10-40% = Blue (low accident risk)
# 40-60% = Yellow (moderate accident risk)
# 60-80% = Orange
# 80-95%+ = Red
# ─────────────────────────────────────────────
HISTORICAL_RISK_WEIGHT = 0.5
PREDICTIVE_RISK_WEIGHT = 0.5

RISK_CATEGORIES = {
    "No Risk":     (0, 10),
    "Low":         (10, 40),
    "Moderate":    (40, 60),
    "High":        (60, 80),
    "Very High":   (80, 101),  # 80-100 (inclusive of 95+)
}

RISK_COLORS = {
    "No Risk":     "#22C55E",   # Green
    "Low":         "#3B82F6",   # Blue
    "Moderate":    "#EAB308",   # Yellow
    "High":        "#F97316",   # Orange
    "Very High":   "#EF4444",   # Red
}

# ─────────────────────────────────────────────
# INTERVENTION SETTINGS
# ─────────────────────────────────────────────
INTERVENTIONS = {
    "street_lights": {
        "name": "Street Lights Installation",
        "description": "Install street lighting on dark road segments",
        "cost_per_km_inr": 2500000,
        "expected_risk_reduction_min": 20,
        "expected_risk_reduction_max": 35,
        "implementation_months": 3,
        "primary_benefit": "Reduces night-time accidents",
        "feature_modifications": {"is_night_risk_factor": 0.65},
    },
    "speed_cameras": {
        "name": "Speed Camera Installation",
        "description": "Install automated speed enforcement cameras",
        "cost_per_km_inr": 1500000,
        "expected_risk_reduction_min": 15,
        "expected_risk_reduction_max": 25,
        "implementation_months": 1,
        "primary_benefit": "Reduces speeding-related accidents",
        "feature_modifications": {"speeding_risk_factor": 0.75},
    },
    "median_barriers": {
        "name": "Median Barrier Installation",
        "description": "Install crash barriers on road medians",
        "cost_per_km_inr": 3500000,
        "expected_risk_reduction_min": 25,
        "expected_risk_reduction_max": 40,
        "implementation_months": 6,
        "primary_benefit": "Prevents head-on collisions",
        "feature_modifications": {"head_on_risk_factor": 0.60},
    },
    "road_widening": {
        "name": "Road Widening",
        "description": "Widen narrow road segments",
        "cost_per_km_inr": 15000000,
        "expected_risk_reduction_min": 10,
        "expected_risk_reduction_max": 20,
        "implementation_months": 12,
        "primary_benefit": "Reduces congestion-related accidents",
        "feature_modifications": {"congestion_risk_factor": 0.80},
    },
    "rumble_strips": {
        "name": "Rumble Strip Installation",
        "description": "Install rumble strips to alert drowsy drivers",
        "cost_per_km_inr": 500000,
        "expected_risk_reduction_min": 10,
        "expected_risk_reduction_max": 20,
        "implementation_months": 1,
        "primary_benefit": "Reduces run-off-road accidents",
        "feature_modifications": {"drowsy_risk_factor": 0.80},
    },
    "signage_improvement": {
        "name": "Road Signage Improvement",
        "description": "Improve road signs and markings",
        "cost_per_km_inr": 800000,
        "expected_risk_reduction_min": 8,
        "expected_risk_reduction_max": 15,
        "implementation_months": 2,
        "primary_benefit": "Reduces confusion and wrong-way driving",
        "feature_modifications": {"visibility_risk_factor": 0.85},
    },
}

# Value of statistical life in India (INR)
VALUE_OF_STATISTICAL_LIFE_INR = 10000000
BASELINE_FATALITIES_PER_HIGH_RISK_SEGMENT = 2.5

# ─────────────────────────────────────────────
# CREATE ALL DIRECTORIES
# ─────────────────────────────────────────────
for d in [
    DATA_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    SHAP_DIR,
    ROAD_NETWORKS_DIR,
    MAPPED_ACCIDENTS_DIR,
    DIGITAL_TWIN_DIR,
    GEOCODE_CACHE_DIR,
]:
    os.makedirs(d, exist_ok=True)
