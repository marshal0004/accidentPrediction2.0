import io
import csv
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter(prefix="/api/predict", tags=["Prediction"])

predictor_instance = None


def get_predictor():
    """Get or initialize the predictor singleton."""
    global predictor_instance
    if predictor_instance is None:
        from ml.predictor import AccidentPredictor

        predictor_instance = AccidentPredictor(dataset_key="primary")
        predictor_instance.load_artifacts()
    return predictor_instance


class PredictionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    Day_of_Week: Optional[str] = "Monday"
    Time_of_Accident: Optional[str] = "12:00"
    Accident_Location: Optional[str] = "Urban"
    Nature_of_Accident: Optional[str] = "Head-on Collision"
    Causes_D1: Optional[str] = "Overspeeding"
    Road_Condition: Optional[str] = "Straight"
    Weather_Conditions: Optional[str] = "Clear"
    Intersection_Type: Optional[str] = "None"
    Vehicle_Type_V1: Optional[str] = "Car"
    Vehicle_Type_V2: Optional[str] = "Truck"
    Number_of_Vehicles: Optional[int] = 2
    model_name: Optional[str] = "XGBoost"


@router.post("")
def predict_severity(request: PredictionRequest):
    """Make a severity prediction for new accident data."""
    predictor = get_predictor()

    if not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run the training pipeline first.",
        )

    input_data = request.dict()
    model_name = input_data.pop("model_name", "XGBoost")

    result = predictor.predict(input_data, model_name)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...), model_name: str = "XGBoost"):
    """Batch prediction — accept CSV upload, return predictions."""
    predictor = get_predictor()

    if not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run the training pipeline first.",
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum 50MB.")

        df = pd.read_csv(io.BytesIO(contents))

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty.")

        if len(df) > 10000:
            raise HTTPException(status_code=400, detail="Maximum 10,000 rows allowed.")

        result = predictor.predict_batch(df, model_name)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/models")
def get_available_models():
    """Return list of available models for prediction."""
    predictor = get_predictor()
    return {"models": predictor.get_available_models(), "loaded": predictor.loaded}
