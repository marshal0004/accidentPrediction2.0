"""
Support Vector Machine Classifier for Accident Severity Prediction.

Hyperparameters:
    kernel=rbf, C=1.0, gamma=scale, class_weight=balanced, probability=True

Why SVM:
    - Effective in high-dimensional spaces
    - Works well when number of dimensions exceeds number of samples
    - Kernel trick maps data to higher dimensions for non-linear separation
    - class_weight=balanced handles imbalanced classes
    - probability=True enables predict_proba for ROC/AUC computation
"""

import time
import os
import numpy as np
import joblib
from sklearn.svm import SVC
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


def build_svm():
    """Build and return an SVM classifier with tuned hyperparameters."""
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    return model


def train_svm(X_train, y_train, X_test, y_test, feature_names, label_mapping, save_dir):
    """
    Train SVM model end-to-end.

    Args:
        X_train: Training features (after SMOTE)
        y_train: Training labels (after SMOTE)
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature column names
        label_mapping: Dict mapping encoded labels to original class names
        save_dir: Directory to save model and artifacts

    Returns:
        dict with model, metrics, training_time, cv_scores
    """
    print(f"\n  {'='*50}")
    print(f"  SVM (Support Vector Machine) — Training")
    print(f"  {'='*50}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  NOTE: SVM can be slow on large datasets. Please wait...")

    model = build_svm()

    # --- Train ---
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.2f} seconds")

    # --- Cross Validation ---
    print(f"  Running {CV_FOLDS}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=1
        )
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
    except Exception as e:
        print(f"  [WARNING] CV failed for SVM: {e}")
        cv_scores = np.array([0.0])
        cv_mean = 0.0
        cv_std = 0.0
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

    print(f"\n  --- SVM Results ---")
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

    # --- Save Model ---
    model_path = os.path.join(save_dir, "SVM_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n  [SAVED] SVM model → {model_path}")

    # --- Return Results ---
    result = {
        "model_name": "SVM",
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
        "feature_importances": [],
        "model_path": model_path,
    }

    return result
