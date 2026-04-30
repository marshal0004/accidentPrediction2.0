"""
Train a single model standalone.

Usage:
    python train_single_model.py RandomForest
    python train_single_model.py XGBoost
    python train_single_model.py GradientBoosting
    python train_single_model.py SVM
    python train_single_model.py LogisticRegression
    python train_single_model.py all
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUTS_DIR, MODELS_DIR
from ml.data_loader import load_all_datasets, generate_eda_summary
from ml.preprocessor import preprocess_dataset, scale_features
from ml.trainer import apply_smote
from ml.evaluator import evaluate_all_models, generate_evaluation_plots
from ml.shap_analyzer import compute_shap_for_model
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE

AVAILABLE_MODELS = {
    "RandomForest": "ml.models.random_forest",
    "XGBoost": "ml.models.xgboost_model",
    "GradientBoosting": "ml.models.gradient_boosting",
    "SVM": "ml.models.svm_model",
    "LogisticRegression": "ml.models.logistic_regression",
}


def train_single(model_name, run_shap=True, dataset_key="primary"):
    """Train a single model end-to-end."""

    if model_name not in AVAILABLE_MODELS:
        print(f"\n  [ERROR] Unknown model: '{model_name}'")
        print(f"  Available models: {list(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  SINGLE MODEL TRAINING: {model_name}")
    print(f"{'='*60}")

    # --- Load dataset ---
    print(f"\n  [1/5] Loading dataset...")
    datasets = load_all_datasets()
    if not datasets:
        print("  [FATAL] No datasets found!")
        sys.exit(1)

    ds = datasets.get(dataset_key)
    if ds is None:
        dataset_key = list(datasets.keys())[0]
        ds = datasets[dataset_key]
        print(f"  [INFO] Using dataset: {dataset_key}")

    generate_eda_summary(datasets)

    # --- Preprocess ---
    print(f"\n  [2/5] Preprocessing...")
    df = ds["dataframe"]
    target = ds["target_column"]
    roles = ds["roles"]

    X, y, feature_names, le, label_mapping = preprocess_dataset(df, target, roles, dataset_key)

    # --- Split ---
    print(f"\n  [3/5] Splitting and applying SMOTE...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_names)
    X_train_smote, y_train_smote, smote_method = apply_smote(X_train_scaled, y_train)

    # --- Import and train the specific model ---
    print(f"\n  [4/5] Training {model_name}...")
    save_dir = os.path.join(MODELS_DIR, dataset_key)
    os.makedirs(save_dir, exist_ok=True)

    if model_name == "RandomForest":
        from ml.models.random_forest import train_random_forest
        result = train_random_forest(
            X_train_smote, y_train_smote, X_test_scaled, y_test,
            feature_names, label_mapping, save_dir
        )
    elif model_name == "XGBoost":
        from ml.models.xgboost_model import train_xgboost
        result = train_xgboost(
            X_train_smote, y_train_smote, X_test_scaled, y_test,
            feature_names, label_mapping, save_dir
        )
    elif model_name == "GradientBoosting":
        from ml.models.gradient_boosting import train_gradient_boosting
        result = train_gradient_boosting(
            X_train_smote, y_train_smote, X_test_scaled, y_test,
            feature_names, label_mapping, save_dir
        )
    elif model_name == "SVM":
        from ml.models.svm_model import train_svm
        result = train_svm(
            X_train_smote, y_train_smote, X_test_scaled, y_test,
            feature_names, label_mapping, save_dir
        )
    elif model_name == "LogisticRegression":
        from ml.models.logistic_regression import train_logistic_regression
        result = train_logistic_regression(
            X_train_smote, y_train_smote, X_test_scaled, y_test,
            feature_names, label_mapping, save_dir
        )

    # --- SHAP ---
    if run_shap:
        print(f"\n  [5/5] SHAP Analysis for {model_name}...")
        shap_result = compute_shap_for_model(
            result["model"], model_name, X_test_scaled,
            feature_names, label_mapping, dataset_key
        )
    else:
        print(f"\n  [5/5] Skipping SHAP (use --shap to enable)")

    # --- Summary ---
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  {model_name} TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Time:          {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Accuracy:      {result['accuracy']:.4f}")
    print(f"  F1 (Weighted): {result['f1_weighted']:.4f}")
    print(f"  F1 (Macro):    {result['f1_macro']:.4f}")
    print(f"  ROC-AUC:       {result['roc_auc']:.4f}")
    print(f"  MCC:           {result['mcc']:.4f}")
    print(f"  CV Mean:       {result['cv_mean']:.4f} +/- {result['cv_std']:.4f}")
    print(f"  Model saved:   {result['model_path']}")

    # --- Save individual result JSON ---
    result_json = {k: v for k, v in result.items() if k != "model"}
    result_path = os.path.join(OUTPUTS_DIR, f"result_{model_name}_{dataset_key}.json")
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    print(f"  Result JSON:   {result_path}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"\n  Usage: python train_single_model.py <ModelName> [--no-shap]")
        print(f"\n  Available models:")
        for name in AVAILABLE_MODELS:
            print(f"    - {name}")
        print(f"    - all (trains all 5 models)")
        print(f"\n  Examples:")
        print(f"    python train_single_model.py RandomForest")
        print(f"    python train_single_model.py XGBoost --no-shap")
        print(f"    python train_single_model.py SVM")
        print(f"    python train_single_model.py all")
        sys.exit(0)

    model_name = sys.argv[1]
    run_shap = "--no-shap" not in sys.argv

    if model_name == "all":
        print(f"\n  Training ALL models one by one...")
        for name in AVAILABLE_MODELS:
            train_single(name, run_shap=run_shap)
    else:
        train_single(model_name, run_shap=run_shap)
