import os
import json
from fastapi import APIRouter, HTTPException
from config import OUTPUTS_DIR, PLOTS_DIR

router = APIRouter(prefix="/api/eda", tags=["EDA"])


def load_eda_summary():
    """Load EDA summary from saved JSON."""
    path = os.path.join(OUTPUTS_DIR, "eda_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@router.get("/summary")
def get_eda_summary():
    """Return EDA summary statistics for all loaded datasets."""
    summary = load_eda_summary()
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="EDA summary not found. Run training pipeline first.",
        )
    return summary


@router.get("/summary/{dataset_key}")
def get_eda_summary_by_dataset(dataset_key: str):
    """Return EDA summary for a specific dataset."""
    summary = load_eda_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="EDA summary not found.")
    if dataset_key not in summary:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_key}' not found in summary."
        )
    return summary[dataset_key]


@router.get("/charts/{chart_name}")
def get_chart_data(chart_name: str, dataset_key: str = "primary"):
    """Return chart data as JSON for frontend rendering."""
    summary = load_eda_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="EDA data not available.")

    ds = summary.get(dataset_key)
    if ds is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_key}' not found."
        )

    severity_colors = ["#EF4444", "#F97316", "#F59E0B", "#10B981"]

    if chart_name == "class_distribution":
        dist = ds.get("class_distribution", {})
        labels = list(dist.keys())
        values = list(dist.values())
        colors = severity_colors[: len(labels)]
        return {
            "chart_type": "donut",
            "title": "Accident Severity Distribution",
            "labels": labels,
            "values": values,
            "colors": colors,
        }

    chart_data_path = os.path.join(
        OUTPUTS_DIR, f"chart_data_{chart_name}_{dataset_key}.json"
    )
    if os.path.exists(chart_data_path):
        with open(chart_data_path) as f:
            return json.load(f)

    if chart_name == "accidents_by_hour":
        return {
            "chart_type": "area",
            "title": "Accidents by Hour of Day",
            "labels": list(range(24)),
            "values": [0] * 24,
            "message": "Time-based data will be available after training pipeline runs with time features.",
        }

    if chart_name == "accidents_by_day":
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        return {
            "chart_type": "bar",
            "title": "Accidents by Day of Week",
            "labels": days,
            "values": [0] * 7,
            "message": "Day-based data will be available after training pipeline runs.",
        }

    if chart_name == "accidents_by_weather":
        return {
            "chart_type": "grouped_bar",
            "title": "Weather Impact on Severity",
            "labels": [],
            "datasets": [],
            "message": "Weather data will be available after training pipeline runs.",
        }

    if chart_name == "accidents_by_vehicle":
        return {
            "chart_type": "pie",
            "title": "Vehicle Type Distribution",
            "labels": [],
            "values": [],
            "message": "Vehicle data will be available after training pipeline runs.",
        }

    if chart_name == "severity_by_cause":
        return {
            "chart_type": "horizontal_bar",
            "title": "Top Accident Causes by Severity",
            "labels": [],
            "datasets": [],
            "message": "Cause data will be available after training pipeline runs.",
        }

    if chart_name == "monthly_trend":
        return {
            "chart_type": "line",
            "title": "Monthly Accident Trend",
            "labels": [],
            "datasets": [],
            "message": "Monthly trend data will be available after training pipeline runs.",
        }

    if chart_name == "correlation_matrix":
        return {
            "chart_type": "heatmap",
            "title": "Feature Correlation Matrix",
            "labels": [],
            "values": [],
            "message": "Correlation data will be available after training pipeline runs.",
        }

    raise HTTPException(status_code=404, detail=f"Chart '{chart_name}' not found.")


@router.get("/plots/{plot_name}")
def get_plot_image(plot_name: str):
    """Return a saved plot as base64 encoded image."""
    import base64

    plot_path = os.path.join(PLOTS_DIR, f"{plot_name}.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found.")

    with open(plot_path, "rb") as f:
        img_data = f.read()

    b64 = base64.b64encode(img_data).decode("utf-8")
    return {"image_base64": f"data:image/png;base64,{b64}", "plot_name": plot_name}
