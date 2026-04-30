"""
Gradient Boosting Classifier for Accident Severity Prediction.

Hyperparameters:
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, min_samples_split=5, min_samples_leaf=2

Why Gradient Boosting:
    - Sequential ensemble that corrects errors from previous trees
    - More conservative than XGBoost (no regularization params)
    - Good baseline for comparing against XGBoost/LightGBM
    - Scikit-learn native — no extra library dependency
"""

import time
import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

RANDOM_STATE = 42
CV_FOLDS = 5


def build_gradient_boosting():
    """Build and return a Gradient Boosting classifier with tuned hyperparameters."""
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    return model


def train_gradient_boosting(
    X_train, y_train, X_test, y_test, feature_names, label_mapping, save_dir
):
    """
    Train Gradient Boosting model end-to-end.

    Args:
        X_train: Training features (after SMOTE)
        y_train: Training labels (after SMOTE)
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature column names
        label_mapping: Dict mapping encoded labels to original class names
        save_dir: Directory to save model and artifacts

    Returns:
        dict with model, metrics, training_time, cv_scores, feature_importances
    """
    print(f"\n  {'='*50}")
    print(f"  GRADIENT BOOSTING — Training")
    print(f"  {'='*50}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")

    model = build_gradient_boosting()

    # --- Train ---
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.2f} seconds")

    # --- Cross Validation ---
    print(f"  Running {CV_FOLDS}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1
    )
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"  CV F1 (weighted): {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  CV fold scores: {[round(s, 4) for s in cv_scores]}")

    # --- Predict ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        ll = log_loss(y_test, y_proba)
    except Exception:
        ll = 0.0

    try:
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            roc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            y_test_bin = label_binarize(y_test, classes=sorted(np.unique(y_test)))
            roc = roc_auc_score(
                y_test_bin, y_proba, average="weighted", multi_class="ovr"
            )
    except Exception:
        roc = 0.0

    print(f"\n  --- Gradient Boosting Results ---")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Precision (W):     {prec_w:.4f}")
    print(f"  Recall (W):        {rec_w:.4f}")
    print(f"  F1 (Weighted):     {f1_w:.4f}")
    print(f"  F1 (Macro):        {f1_m:.4f}")
    print(f"  ROC-AUC (W):       {roc:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    print(f"  MCC:               {mcc:.4f}")
    print(f"  Log Loss:          {ll:.4f}")

    # --- Classification Report ---
    class_names = [str(label_mapping.get(str(c), c)) for c in sorted(np.unique(y_test))]
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    print(f"\n  Classification Report:\n{report}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred).tolist()
    cm_array = np.array(cm)
    cm_normalized = []
    for row in cm_array:
        row_sum = row.sum()
        if row_sum > 0:
            cm_normalized.append((row / row_sum).tolist())
        else:
            cm_normalized.append(row.tolist())

    # --- ROC Data ---
    roc_data = {}
    classes = sorted(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=classes)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    for i, cls in enumerate(classes):
        if i < y_proba.shape[1] and i < y_test_bin.shape[1]:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            cls_name = str(label_mapping.get(str(cls), cls))
            roc_data[cls_name] = {
                "fpr": [round(float(x), 4) for x in fpr[:: max(1, len(fpr) // 50)]],
                "tpr": [round(float(x), 4) for x in tpr[:: max(1, len(tpr) // 50)]],
                "auc": round(float(roc_auc_val), 4),
            }

    # --- Feature Importance ---
    importances = model.feature_importances_
    top_n = min(20, len(importances))
    sorted_idx = np.argsort(importances)[-top_n:][::-1]
    feature_importances = []
    for idx in sorted_idx:
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        feature_importances.append(
            {"name": fname, "importance": round(float(importances[idx]), 6)}
        )

    print(f"\n  Top 10 Feature Importances:")
    for fi in feature_importances[:10]:
        print(f"    {fi['name']}: {fi['importance']:.6f}")

    # --- Save Model ---
    model_path = os.path.join(save_dir, "GradientBoosting_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n  [SAVED] Gradient Boosting model → {model_path}")

    # --- Return Results ---
    result = {
        "model_name": "GradientBoosting",
        "model": model,
        "training_time": round(training_time, 2),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted": round(rec_w, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "roc_auc": round(roc, 4),
        "cohens_kappa": round(kappa, 4),
        "mcc": round(mcc, 4),
        "log_loss": round(ll, 4),
        "confusion_matrix": cm,
        "normalized_confusion_matrix": cm_normalized,
        "roc_data": roc_data,
        "class_labels": class_names,
        "feature_importances": feature_importances,
        "model_path": model_path,
    }

    return result
