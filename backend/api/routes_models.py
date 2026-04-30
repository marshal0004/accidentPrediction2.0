import os
import json
from fastapi import APIRouter, HTTPException
from config import OUTPUTS_DIR

router = APIRouter(prefix="/api/models", tags=["Models"])


def load_model_comparison():
    """Load model comparison results from saved JSON."""
    path = os.path.join(OUTPUTS_DIR, "model_comparison.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@router.get("/comparison")
def get_model_comparison():
    """Return model comparison table with all metrics."""
    comparison = load_model_comparison()
    if comparison is None:
        raise HTTPException(
            status_code=404,
            detail="Model comparison not found. Run training pipeline first.",
        )
    return comparison


@router.get("/{model_name}/confusion-matrix")
def get_confusion_matrix(model_name: str):
    """Return confusion matrix data for a specific model."""
    comparison = load_model_comparison()
    if comparison is None:
        raise HTTPException(status_code=404, detail="Model results not found.")

    for model in comparison.get("models", []):
        if model["model_name"] == model_name:
            return {
                "model": model_name,
                "matrix": model.get("confusion_matrix", []),
                "normalized_matrix": model.get("normalized_confusion_matrix", []),
                "labels": model.get("class_labels", []),
            }

    available = [m["model_name"] for m in comparison.get("models", [])]
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_name}' not found. Available: {available}",
    )


@router.get("/{model_name}/roc-data")
def get_roc_data(model_name: str):
    """Return ROC curve data points for a specific model."""
    comparison = load_model_comparison()
    if comparison is None:
        raise HTTPException(status_code=404, detail="Model results not found.")

    for model in comparison.get("models", []):
        if model["model_name"] == model_name:
            return {"model": model_name, "classes": model.get("roc_data", {})}

    available = [m["model_name"] for m in comparison.get("models", [])]
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_name}' not found. Available: {available}",
    )


@router.get("/{model_name}/metrics")
def get_model_metrics(model_name: str):
    """Return detailed metrics for a specific model."""
    comparison = load_model_comparison()
    if comparison is None:
        raise HTTPException(status_code=404, detail="Model results not found.")

    for model in comparison.get("models", []):
        if model["model_name"] == model_name:
            return model

    available = [m["model_name"] for m in comparison.get("models", [])]
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_name}' not found. Available: {available}",
    )


@router.get("/best")
def get_best_model():
    """Return info about the best performing model."""
    comparison = load_model_comparison()
    if comparison is None:
        raise HTTPException(status_code=404, detail="Model results not found.")

    return {
        "best_model": comparison.get("best_model"),
        "best_metric": comparison.get("best_metric"),
        "best_value": comparison.get("best_value"),
    }
