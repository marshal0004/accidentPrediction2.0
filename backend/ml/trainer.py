"""
Model Training Pipeline — orchestrates all 5 models.
Each model lives in its own file under ml/models/.
This file handles: split → SMOTE → call each model trainer → collect results.
"""

import numpy as np
import time
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from config import RANDOM_STATE, TEST_SIZE, MODELS_DIR
from ml.models import MODEL_TRAINERS


def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance on training data only."""
    print(f"\n  [SMOTE] Class distribution BEFORE SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls}: {cnt} samples")

    min_class_count = min(counts)
    k_neighbors = 5 if min_class_count >= 6 else max(1, min_class_count - 1)

    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"\n  [SMOTE] Class distribution AFTER SMOTE:")
        unique_r, counts_r = np.unique(y_resampled, return_counts=True)
        for cls, cnt in zip(unique_r, counts_r):
            print(f"    Class {cls}: {cnt} samples")

        print(f"  [SMOTE] Training set size: {len(X_train)} → {len(X_resampled)}")
        return X_resampled, y_resampled, "SMOTE"

    except Exception as e:
        print(f"  [WARNING] SMOTE failed: {e}. Trying SMOTETomek...")
        try:
            smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
            print(f"  [SMOTETomek] Applied successfully. New size: {len(X_resampled)}")
            return X_resampled, y_resampled, "SMOTETomek"
        except Exception as e2:
            print(f"  [ERROR] SMOTETomek also failed: {e2}. Using original data.")
            return X_train, y_train, "None"


def split_data(X, y):
    """Split data into train and test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n  [SPLIT] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_all_models(X, y, feature_names, label_mapping, dataset_key="primary"):
    """
    Full training pipeline:
        1. Split data (80/20 stratified)
        2. Scale features
        3. Apply SMOTE on training set
        4. Train each of the 5 models using their dedicated files
        5. Collect and return all results

    Args:
        X: Full feature matrix
        y: Full target array
        feature_names: List of feature column names
        label_mapping: Dict mapping encoded labels to class names
        dataset_key: Identifier for the dataset being used

    Returns:
        results: Dict of model_name → result dict (includes model, metrics, etc.)
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Original training labels (before SMOTE)
        y_test: Test labels
        X_train_smote: SMOTE-resampled training features
        y_train_smote: SMOTE-resampled training labels
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING PIPELINE: {dataset_key}")
    print(f"  Models: {list(MODEL_TRAINERS.keys())}")
    print(f"{'='*60}")

    # --- Step 1: Split ---
    X_train, X_test, y_train, y_test = split_data(X, y)

    # --- Step 2: Scale ---
    from ml.preprocessor import scale_features

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, feature_names
    )

    # --- Step 3: SMOTE ---
    X_train_smote, y_train_smote, smote_method = apply_smote(X_train_scaled, y_train)

    # --- Step 4: Train each model ---
    results = {}
    save_dir = os.path.join(MODELS_DIR, dataset_key)
    os.makedirs(save_dir, exist_ok=True)

    for model_name, train_fn in MODEL_TRAINERS.items():
        print(f"\n  {'#'*50}")
        print(f"  Starting: {model_name}")
        print(f"  {'#'*50}")

        try:
            result = train_fn(
                X_train=X_train_smote,
                y_train=y_train_smote,
                X_test=X_test_scaled,
                y_test=y_test,
                feature_names=feature_names,
                label_mapping=label_mapping,
                save_dir=save_dir,
            )
            results[model_name] = result
            print(f"  [OK] {model_name} — F1(W): {result['f1_weighted']:.4f}")

        except Exception as e:
            print(f"  [ERROR] {model_name} training failed: {e}")
            import traceback

            traceback.print_exc()
            results[model_name] = {
                "model_name": model_name,
                "model": None,
                "error": str(e),
                "training_time": 0,
                "cv_scores": [],
                "cv_mean": 0.0,
                "cv_std": 0.0,
                "accuracy": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_weighted": 0.0,
                "f1_macro": 0.0,
                "roc_auc": 0.0,
                "cohens_kappa": 0.0,
                "mcc": 0.0,
                "log_loss": 0.0,
                "confusion_matrix": [],
                "normalized_confusion_matrix": [],
                "roc_data": {},
                "class_labels": [],
                "feature_importances": [],
            }

    # --- Step 5: Summary ---
    successful = {k: v for k, v in results.items() if v.get("model") is not None}
    failed = {k: v for k, v in results.items() if v.get("model") is None}

    print(f"\n  {'='*50}")
    print(f"  TRAINING SUMMARY")
    print(f"  {'='*50}")
    print(f"  Models trained successfully: {len(successful)}/{len(results)}")

    if successful:
        best = max(successful.values(), key=lambda x: x.get("f1_weighted", 0))
        print(f"  Best model: {best['model_name']} (F1-W: {best['f1_weighted']:.4f})")

    if failed:
        print(f"  Failed models: {list(failed.keys())}")
        for name, res in failed.items():
            print(f"    {name}: {res.get('error', 'Unknown error')}")

    # --- Save training summary ---
    training_summary = {
        "dataset": dataset_key,
        "smote_method": smote_method,
        "train_size_original": int(X_train_scaled.shape[0]),
        "train_size_after_smote": int(X_train_smote.shape[0]),
        "test_size": int(X_test_scaled.shape[0]),
        "n_features": int(X_train_scaled.shape[1]),
        "models_trained": list(successful.keys()),
        "models_failed": list(failed.keys()),
        "best_model": best["model_name"] if successful else None,
        "best_f1_weighted": best["f1_weighted"] if successful else 0.0,
    }

    summary_path = os.path.join(MODELS_DIR, f"training_summary_{dataset_key}.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2, default=str)
    print(f"\n  [SAVED] Training summary → {summary_path}")

    return (
        results,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_train_smote,
        y_train_smote,
    )
