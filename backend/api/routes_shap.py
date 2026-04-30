import os
import json
from fastapi import APIRouter, HTTPException
from config import OUTPUTS_DIR, SHAP_DIR

router = APIRouter(prefix="/api/shap", tags=["SHAP"])


@router.get("/feature-importance")
def get_shap_feature_importance(model_name: str = None):
    """Return SHAP feature importance. If model_name given, return for that model."""
    if model_name:
        specific_path = os.path.join(
            SHAP_DIR, f"shap_feature_importance_{model_name}_primary.json"
        )
        if os.path.exists(specific_path):
            with open(specific_path) as f:
                features = json.load(f)
            return {"model": model_name, "features": features}

    combined_path = os.path.join(OUTPUTS_DIR, "shap_feature_importance.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            data = json.load(f)
        return data

    raise HTTPException(
        status_code=404,
        detail="SHAP feature importance not found. Run training pipeline first.",
    )


@router.get("/feature-importance/{model_name}")
def get_shap_for_model(model_name: str, dataset_key: str = "primary"):
    """Return SHAP feature importance for a specific model."""
    path = os.path.join(
        SHAP_DIR, f"shap_feature_importance_{model_name}_{dataset_key}.json"
    )
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"SHAP data not found for model '{model_name}'. Run SHAP analysis first.",
        )

    with open(path) as f:
        features = json.load(f)

    return {"model": model_name, "dataset": dataset_key, "features": features}


@router.get("/summary-plot")
def get_shap_summary_plot(model_name: str = None, dataset_key: str = "primary"):
    """Return SHAP summary plot as base64 PNG."""
    import base64

    if model_name:
        plot_path = os.path.join(
            SHAP_DIR, f"shap_summary_{model_name}_{dataset_key}.png"
        )
    else:
        all_shap_path = os.path.join(SHAP_DIR, f"all_shap_results_{dataset_key}.json")
        if os.path.exists(all_shap_path):
            with open(all_shap_path) as f:
                all_data = json.load(f)
            first_model = list(all_data.keys())[0] if all_data else None
            if first_model:
                plot_path = os.path.join(
                    SHAP_DIR, f"shap_summary_{first_model}_{dataset_key}.png"
                )
            else:
                raise HTTPException(status_code=404, detail="No SHAP data available.")
        else:
            raise HTTPException(status_code=404, detail="SHAP data not found.")

    if not os.path.exists(plot_path):
        raise HTTPException(
            status_code=404, detail=f"SHAP summary plot not found at {plot_path}"
        )

    with open(plot_path, "rb") as f:
        img_data = f.read()

    b64 = base64.b64encode(img_data).decode("utf-8")
    return {"image_base64": f"data:image/png;base64,{b64}", "model": model_name}


@router.get("/bar-plot/{model_name}")
def get_shap_bar_plot(model_name: str, dataset_key: str = "primary"):
    """Return SHAP bar plot as base64 PNG."""
    import base64

    plot_path = os.path.join(SHAP_DIR, f"shap_bar_{model_name}_{dataset_key}.png")
    if not os.path.exists(plot_path):
        raise HTTPException(
            status_code=404, detail=f"SHAP bar plot not found for '{model_name}'."
        )

    with open(plot_path, "rb") as f:
        img_data = f.read()

    b64 = base64.b64encode(img_data).decode("utf-8")
    return {"image_base64": f"data:image/png;base64,{b64}", "model": model_name}


@router.get("/all-models")
def get_all_shap_results(dataset_key: str = "primary"):
    """Return SHAP results summary for all models."""
    path = os.path.join(SHAP_DIR, f"all_shap_results_{dataset_key}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="SHAP results not found.")

    with open(path) as f:
        data = json.load(f)

    return {"dataset": dataset_key, "models": data}
