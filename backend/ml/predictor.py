import os
import json
import numpy as np
import joblib
from config import MODELS_DIR, SHAP_DIR


class AccidentPredictor:
    """Inference engine for accident severity prediction."""

    def __init__(self, dataset_key="primary"):
        self.dataset_key = dataset_key
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.label_mapping = {}
        self.loaded = False

    def load_artifacts(self):
        """Load all trained models and preprocessing artifacts."""
        print(f"\n  Loading prediction artifacts for '{self.dataset_key}'...")

        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  [OK] Scaler loaded")
        else:
            print(f"  [WARNING] Scaler not found at {scaler_path}")

        le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            print(f"  [OK] Label encoder loaded")
        else:
            print(f"  [WARNING] Label encoder not found at {le_path}")

        fi_path = os.path.join(MODELS_DIR, f"feature_info_{self.dataset_key}.json")
        if os.path.exists(fi_path):
            with open(fi_path) as f:
                fi = json.load(f)
            self.feature_names = fi.get("feature_names", [])
            self.label_mapping = fi.get("label_mapping", {})
            print(f"  [OK] Feature info loaded ({len(self.feature_names)} features)")
        else:
            print(f"  [WARNING] Feature info not found at {fi_path}")

        model_dir = os.path.join(MODELS_DIR, self.dataset_key)
        model_names = [
            "RandomForest",
            "XGBoost",
            "GradientBoosting",
            "SVM",
            "LogisticRegression",
        ]

        for name in model_names:
            model_path = os.path.join(model_dir, f"{name}_model.joblib")
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"  [OK] {name} model loaded from {model_path}")
            else:
                old_path = os.path.join(
                    MODELS_DIR, f"{name}_{self.dataset_key}_model.joblib"
                )
                if os.path.exists(old_path):
                    self.models[name] = joblib.load(old_path)
                    print(f"  [OK] {name} model loaded from {old_path}")
                else:
                    print(f"  [SKIP] {name} model not found")

        self.loaded = len(self.models) > 0
        print(
            f"  [{'OK' if self.loaded else 'ERROR'}] {len(self.models)} models loaded"
        )
        return self.loaded

    def prepare_input(self, input_data):
        """Convert raw input dict to feature vector matching training features."""
        feature_vector = np.zeros(len(self.feature_names))

        for i, fname in enumerate(self.feature_names):
            fname_lower = fname.lower()

            for key, value in input_data.items():
                key_lower = key.lower()

                if key_lower == fname_lower:
                    if isinstance(value, (int, float)):
                        feature_vector[i] = value
                    break

                if fname_lower.startswith(key_lower + "_"):
                    suffix = fname[len(key) + 1 :]
                    if (
                        isinstance(value, str)
                        and value.strip().lower() == suffix.lower()
                    ):
                        feature_vector[i] = 1.0
                    break

                if "_freq" in fname_lower and key_lower in fname_lower:
                    feature_vector[i] = 0.05
                    break

        if "Hour" in self.feature_names:
            time_str = input_data.get("Time_of_Accident", "12:00")
            try:
                parts = str(time_str).split(":")
                hour = int(parts[0])
                idx = self.feature_names.index("Hour")
                feature_vector[idx] = hour
            except (ValueError, IndexError):
                pass

        if "Is_Night" in self.feature_names:
            try:
                hour = feature_vector[self.feature_names.index("Hour")]
                idx = self.feature_names.index("Is_Night")
                feature_vector[idx] = 1.0 if (hour >= 21 or hour < 6) else 0.0
            except (ValueError, IndexError):
                pass

        if "Is_Weekend" in self.feature_names:
            day = input_data.get("Day_of_Week", "Monday")
            weekend_days = ["saturday", "sunday"]
            try:
                idx = self.feature_names.index("Is_Weekend")
                feature_vector[idx] = 1.0 if str(day).lower() in weekend_days else 0.0
            except (ValueError, IndexError):
                pass

        if "Number_of_Vehicles" in self.feature_names:
            try:
                idx = self.feature_names.index("Number_of_Vehicles")
                feature_vector[idx] = float(input_data.get("Number_of_Vehicles", 2))
            except (ValueError, IndexError):
                pass

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector = feature_vector.reshape(1, -1)

        return feature_vector

    def predict(self, input_data, model_name="XGBoost"):
        """Make a prediction using specified model."""
        if not self.loaded:
            return {"error": "Models not loaded. Run training first."}

        if model_name not in self.models:
            available = list(self.models.keys())
            if available:
                model_name = available[0]
                print(f"  [INFO] Requested model not found. Using {model_name}")
            else:
                return {"error": "No models available."}

        model = self.models[model_name]
        X = self.prepare_input(input_data)

        prediction_code = int(model.predict(X)[0])

        prediction_label = self.label_mapping.get(
            str(prediction_code), str(prediction_code)
        )

        probabilities = {}
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            for i, p in enumerate(proba):
                class_label = self.label_mapping.get(str(i), str(i))
                probabilities[class_label] = round(float(p), 4)
            confidence = round(float(max(proba)), 4)
        else:
            confidence = 1.0
            probabilities[prediction_label] = 1.0

        top_risk_factors = []
        shap_fi_path = os.path.join(
            SHAP_DIR, f"shap_feature_importance_{model_name}_{self.dataset_key}.json"
        )
        if os.path.exists(shap_fi_path):
            with open(shap_fi_path) as f:
                shap_features = json.load(f)
            for feat in shap_features[:3]:
                top_risk_factors.append(
                    {"feature": feat["name"], "contribution": feat["importance"]}
                )

        result = {
            "prediction": prediction_label,
            "prediction_code": prediction_code,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_risk_factors": top_risk_factors,
            "model_used": model_name,
        }

        return result

    def predict_batch(self, df, model_name="XGBoost"):
        """Make predictions for a batch of records."""
        predictions = []
        for idx, row in df.iterrows():
            input_data = row.to_dict()
            result = self.predict(input_data, model_name)
            result["row"] = idx + 1
            predictions.append(result)

        summary = {}
        for p in predictions:
            label = p.get("prediction", "Unknown")
            summary[label] = summary.get(label, 0) + 1

        return {
            "total_records": len(predictions),
            "predictions": predictions,
            "summary": summary,
        }

    def get_available_models(self):
        """Return list of loaded model names."""
        return list(self.models.keys())

    def get_feature_names(self):
        """Return list of feature names used by models."""
        return self.feature_names

    def get_label_mapping(self):
        """Return label mapping dict."""
        return self.label_mapping
