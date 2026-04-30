"""
Delhi Accident Severity Training Pipeline
==========================================

Production-grade ML training pipeline specifically for Delhi accident data.
Loads the segment mapping produced by delhi_data_mapper, engineers
Delhi-specific features, trains 5 models with hyperparameter tuning,
and saves everything to outputs/models/delhi/.

Usage:
    # Standalone
    python -m ml.delhi_trainer

    # From code
    from ml.delhi_trainer import DelhiTrainer
    trainer = DelhiTrainer()
    results = trainer.run()
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import (
    CV_FOLDS,
    MAPPED_ACCIDENTS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("delhi_trainer")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DELHI_SEGMENT_MAPPING_PATH = os.path.join(
    MAPPED_ACCIDENTS_DIR, "delhi", "segment_mapping.json"
)
DELHI_MODELS_DIR = os.path.join(MODELS_DIR, "delhi")

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────
FEATURE_COLUMNS: List[str] = [
    "total_accidents",
    "fatal_count",
    "grievous_count",
    "minor_count",
    "fatal_rate",
    "accident_density",
    "weighted_severity",
    "length_m",
    "num_years",
    "year_span",
    "day_accident_ratio",
    "is_highway",
    "is_urban",
    "is_intersection",
    "is_virtual",
]

# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACES (Delhi-specific)
# ─────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
}

SVM_PARAMS = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["rbf"],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────


def _load_segment_mapping(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the Delhi segment mapping JSON produced by delhi_data_mapper.

    Args:
        path: Override path to segment_mapping.json. Defaults to config-based path.

    Returns:
        Dict mapping segment_id → segment data.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError: If the mapping file is empty or invalid.
    """
    mapping_path = path or DELHI_SEGMENT_MAPPING_PATH
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Delhi segment mapping not found at {mapping_path}. "
            "Run delhi_data_mapper first."
        )
    with open(mapping_path, "r") as fh:
        data = json.load(fh)
    if not data:
        raise ValueError("Segment mapping file is empty.")
    logger.info("Loaded segment mapping: %d segments", len(data))
    return data


def _extract_features(segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract ML-ready features from a single segment record.

    Args:
        segment: One segment dict from the segment mapping.

    Returns:
        Feature dict with keys from FEATURE_COLUMNS, or None if invalid.
    """
    try:
        total = segment.get("total_accidents", 0)
        if total <= 0:
            return None

        sev_dist = segment.get("severity_distribution", {})
        fatal_count = int(sev_dist.get("Fatal", 0))
        grievous_count = int(sev_dist.get("Grievous", 0))
        minor_count = int(sev_dist.get("Minor", 0))
        fatal_rate = float(segment.get("fatal_rate", 0.0))

        length_m = float(segment.get("length_m", 100.0))
        if length_m <= 0:
            length_m = 100.0

        accident_density = total / (length_m / 1000.0)  # per km

        # Weighted severity: Fatal*10 + Grievous*5 + Minor*2
        weighted_severity = (
            (fatal_count * 10 + grievous_count * 5 + minor_count * 2) / total
        )

        year_dist = segment.get("year_distribution", {})
        num_years = len(year_dist) if year_dist else 1
        year_values = [int(y) for y in year_dist.keys() if y.isdigit()]
        year_span = (max(year_values) - min(year_values)) if len(year_values) >= 2 else 0

        time_dist = segment.get("time_distribution", {})
        day_accidents = (
            int(time_dist.get("Morning", 0))
            + int(time_dist.get("Afternoon", 0))
        )
        night_accidents = (
            int(time_dist.get("Evening", 0))
            + int(time_dist.get("Night", 0))
        )
        total_time = day_accidents + night_accidents
        day_accident_ratio = day_accidents / total_time if total_time > 0 else 0.5

        road_type = str(segment.get("road_type", "")).lower()
        is_highway = 1 if any(
            kw in road_type for kw in ["trunk", "motorway", "primary"]
        ) else 0
        is_urban = 1 if any(
            kw in road_type for kw in ["residential", "tertiary", "living_street"]
        ) else 0
        is_intersection = 1 if "intersection" in road_type else 0

        # Virtual segment flag: 1 if segment was created from GPS points
        # (not mapped to a real OSM road segment), 0 for real segments
        is_virtual = 1 if bool(segment.get("is_virtual", False)) else 0

        return {
            "total_accidents": total,
            "fatal_count": fatal_count,
            "grievous_count": grievous_count,
            "minor_count": minor_count,
            "fatal_rate": fatal_rate,
            "accident_density": round(accident_density, 4),
            "weighted_severity": round(weighted_severity, 4),
            "length_m": length_m,
            "num_years": num_years,
            "year_span": year_span,
            "day_accident_ratio": round(day_accident_ratio, 4),
            "is_highway": is_highway,
            "is_urban": is_urban,
            "is_intersection": is_intersection,
            "is_virtual": is_virtual,
        }
    except Exception as exc:
        logger.warning("Feature extraction failed for segment %s: %s",
                        segment.get("segment_id", "?"), exc)
        return None


def _assign_risk_class(segment: Dict[str, Any]) -> Optional[str]:
    """Assign risk severity class based on composite risk score.

    Rules:
        - High: fatal_rate >= 0.2 OR total_accidents >= 20
        - Medium: fatal_rate >= 0.05 OR total_accidents >= 5
        - Low: everything else

    Args:
        segment: Segment dict from the mapping.

    Returns:
        'High', 'Medium', or 'Low', or None if data invalid.
    """
    total = segment.get("total_accidents", 0)
    fatal_rate = float(segment.get("fatal_rate", 0.0))
    if total <= 0:
        return None
    if fatal_rate >= 0.2 or total >= 20:
        return "High"
    if fatal_rate >= 0.05 or total >= 5:
        return "Medium"
    return "Low"


def _apply_smote(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Apply SMOTE for class imbalance on training data.

    Falls back to SMOTETomek if SMOTE fails, or returns original data.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Tuple of (X_resampled, y_resampled, method_name).
    """
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info("Class distribution BEFORE SMOTE:")
    for cls, cnt in zip(unique, counts):
        logger.info("  Class %s: %d samples", cls, cnt)

    min_count = int(counts.min())
    k_neighbors = 5 if min_count >= 6 else max(1, min_count - 1)

    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        method = "SMOTE"
        logger.info("SMOTE applied: %d → %d samples", len(X_train), len(X_res))
    except Exception as exc:
        logger.warning("SMOTE failed (%s), trying SMOTETomek...", exc)
        try:
            smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
            X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
            method = "SMOTETomek"
            logger.info("SMOTETomek applied: %d → %d samples", len(X_train), len(X_res))
        except Exception as exc2:
            logger.error("SMOTETomek also failed (%s), using original data.", exc2)
            return X_train, y_train, "None"

    unique_r, counts_r = np.unique(y_res, return_counts=True)
    logger.info("Class distribution AFTER %s:", method)
    for cls, cnt in zip(unique_r, counts_r):
        logger.info("  Class %s: %d samples", cls, cnt)

    return X_res, y_res, method


def _compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (or None).
        label_mapping: Mapping from encoded label to class name.

    Returns:
        Dict of metric names to values.
    """
    acc = accuracy_score(y_test, y_pred)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # ROC-AUC
    roc = 0.0
    if y_proba is not None:
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

    # Log Loss
    ll = 0.0
    if y_proba is not None:
        try:
            ll = log_loss(y_test, y_proba)
        except Exception:
            ll = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()
    cm_array = np.array(cm)
    cm_normalized = []
    for row in cm_array:
        row_sum = row.sum()
        cm_normalized.append((row / row_sum).tolist() if row_sum > 0 else row.tolist())

    # Classification report
    class_names = [
        str(label_mapping.get(str(c), c)) for c in sorted(np.unique(y_test))
    ]
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )

    return {
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
        "classification_report": report,
        "class_labels": class_names,
    }


# ─────────────────────────────────────────────
# MAIN TRAINER CLASS
# ─────────────────────────────────────────────


class DelhiTrainer:
    """Production-grade ML training pipeline for Delhi accident data.

    Loads the segment mapping from delhi_data_mapper, engineers features,
    trains 5 models with hyperparameter tuning, evaluates, and persists
    all artifacts.

    Example::

        trainer = DelhiTrainer()
        results = trainer.run()
        # or
        results = trainer.run(segment_mapping_path="/custom/path.json")
    """

    def __init__(
        self,
        segment_mapping_path: Optional[str] = None,
        random_state: int = RANDOM_STATE,
        test_size: float = TEST_SIZE,
        cv_folds: int = CV_FOLDS,
    ) -> None:
        self.segment_mapping_path = segment_mapping_path or DELHI_SEGMENT_MAPPING_PATH
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds

        # Output directories
        self.models_dir = DELHI_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

        # Artifacts (populated during training)
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = list(FEATURE_COLUMNS)
        self.label_mapping: Dict[str, str] = {}
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.training_summary: Dict[str, Any] = {}

    # ───────────────────────────────────────
    # DATA LOADING & FEATURE ENGINEERING
    # ───────────────────────────────────────

    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load segment mapping and engineer features into a DataFrame.

        Args:
            path: Override path to segment_mapping.json.

        Returns:
            DataFrame with feature columns + 'risk_class' target.

        Raises:
            FileNotFoundError: If mapping file missing.
            ValueError: If no valid segments after feature extraction.
        """
        mapping = _load_segment_mapping(path or self.segment_mapping_path)

        rows: List[Dict[str, Any]] = []
        skipped = 0
        for seg_id, seg_data in mapping.items():
            features = _extract_features(seg_data)
            risk_class = _assign_risk_class(seg_data)
            if features is None or risk_class is None:
                skipped += 1
                continue
            features["segment_id"] = seg_id
            features["risk_class"] = risk_class
            rows.append(features)

        if not rows:
            raise ValueError(
                f"No valid segments after feature extraction "
                f"(skipped {skipped}). Check segment mapping data."
            )

        df = pd.DataFrame(rows)
        logger.info(
            "Feature matrix: %d rows × %d features (%d segments skipped)",
            len(df), len(self.feature_names), skipped,
        )

        # Log class distribution
        class_counts = df["risk_class"].value_counts().to_dict()
        logger.info("Risk class distribution: %s", class_counts)

        return df

    # ───────────────────────────────────────
    # PREPROCESSING
    # ───────────────────────────────────────

    def preprocess(
        self, df: pd.DataFrame
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, StandardScaler, LabelEncoder, Dict[str, str],
    ]:
        """Full preprocessing: split, scale, SMOTE.

        Args:
            df: DataFrame from load_data().

        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test,
                      X_train_smote, y_train_smote, scaler, label_encoder,
                      label_mapping).
        """
        # Validate input
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "risk_class" not in df.columns:
            raise ValueError("DataFrame missing 'risk_class' column.")

        X = df[self.feature_names].values.astype(np.float64)
        y_raw = df["risk_class"].values

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        self.label_encoder = le
        self.label_mapping = {
            str(code): name for code, name in zip(le.transform(le.classes_), le.classes_)
        }
        logger.info("Label mapping: %s", self.label_mapping)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        logger.info(
            "Split: train=%d, test=%d", X_train.shape[0], X_test.shape[0],
        )

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scaler = scaler
        logger.info("StandardScaler fitted on %d features", X_train_scaled.shape[1])

        # SMOTE on training set only
        X_train_smote, y_train_smote, smote_method = _apply_smote(
            X_train_scaled, y_train
        )
        logger.info("Balancing method used: %s", smote_method)

        # Save scaler & label encoder immediately
        joblib.dump(scaler, os.path.join(self.models_dir, "scaler.joblib"))
        joblib.dump(le, os.path.join(self.models_dir, "label_encoder.joblib"))
        logger.info("Scaler and label encoder saved to %s", self.models_dir)

        # Save feature info
        feature_info = {
            "feature_names": self.feature_names,
            "label_mapping": self.label_mapping,
            "n_features": len(self.feature_names),
            "n_classes": len(self.label_mapping),
            "dataset": "delhi",
            "source": "segment_mapping",
        }
        with open(os.path.join(self.models_dir, "feature_info.json"), "w") as fh:
            json.dump(feature_info, fh, indent=2)
        logger.info("Feature info saved")

        return (
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            X_train_smote, y_train_smote,
            scaler, le, self.label_mapping,
        )

    # ───────────────────────────────────────
    # INDIVIDUAL MODEL TRAINERS
    # ───────────────────────────────────────

    def _train_xgboost(
        self,
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train XGBoost with RandomizedSearchCV hyperparameter tuning."""
        logger.info("=" * 50)
        logger.info("Training XGBoost with hyperparameter tuning")
        logger.info("=" * 50)

        start = time.time()
        try:
            base_model = XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )
            search = RandomizedSearchCV(
                base_model,
                XGBOOST_PARAMS,
                n_iter=20,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="f1_weighted",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train_smote, y_train_smote)
            model = search.best_estimator_
            best_params = search.best_params_
            logger.info("XGBoost best params: %s", best_params)

            # Cross-validation on best model
            skf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            cv_scores = cross_val_score(
                model, X_train_smote, y_train_smote, cv=skf, scoring="f1_weighted", n_jobs=-1
            )

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            metrics = _compute_metrics(y_test, y_pred, y_proba, self.label_mapping)
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_std"] = round(float(cv_scores.std()), 4)
            metrics["best_params"] = {str(k): str(v) for k, v in best_params.items()}
            metrics["training_time"] = round(time.time() - start, 2)

            # Feature importances
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            metrics["feature_importances"] = [
                {
                    "name": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                    "importance": round(float(importances[i]), 6),
                }
                for i in sorted_idx
            ]

            # Save model
            model_path = os.path.join(self.models_dir, "XGBoost_model.joblib")
            joblib.dump(model, model_path)

            metrics["model_name"] = "XGBoost"
            metrics["model_path"] = model_path
            self.models["XGBoost"] = model

            logger.info(
                "XGBoost done — Acc: %.4f | F1W: %.4f | Time: %.2fs",
                metrics["accuracy"], metrics["f1_weighted"], metrics["training_time"],
            )
            return metrics

        except Exception as exc:
            logger.error("XGBoost training failed: %s", exc)
            traceback.print_exc()
            return self._empty_result("XGBoost", time.time() - start, str(exc))

    def _train_gradient_boosting(
        self,
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train GradientBoosting with RandomizedSearchCV."""
        logger.info("=" * 50)
        logger.info("Training GradientBoosting with hyperparameter tuning")
        logger.info("=" * 50)

        start = time.time()
        try:
            base_model = GradientBoostingClassifier(random_state=self.random_state)
            search = RandomizedSearchCV(
                base_model,
                GRADIENT_BOOSTING_PARAMS,
                n_iter=15,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="f1_weighted",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train_smote, y_train_smote)
            model = search.best_estimator_
            best_params = search.best_params_
            logger.info("GradientBoosting best params: %s", best_params)

            skf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            cv_scores = cross_val_score(
                model, X_train_smote, y_train_smote, cv=skf, scoring="f1_weighted", n_jobs=-1
            )

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            metrics = _compute_metrics(y_test, y_pred, y_proba, self.label_mapping)
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_std"] = round(float(cv_scores.std()), 4)
            metrics["best_params"] = {str(k): str(v) for k, v in best_params.items()}
            metrics["training_time"] = round(time.time() - start, 2)

            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            metrics["feature_importances"] = [
                {
                    "name": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                    "importance": round(float(importances[i]), 6),
                }
                for i in sorted_idx
            ]

            model_path = os.path.join(self.models_dir, "GradientBoosting_model.joblib")
            joblib.dump(model, model_path)

            metrics["model_name"] = "GradientBoosting"
            metrics["model_path"] = model_path
            self.models["GradientBoosting"] = model

            logger.info(
                "GradientBoosting done — Acc: %.4f | F1W: %.4f | Time: %.2fs",
                metrics["accuracy"], metrics["f1_weighted"], metrics["training_time"],
            )
            return metrics

        except Exception as exc:
            logger.error("GradientBoosting training failed: %s", exc)
            traceback.print_exc()
            return self._empty_result("GradientBoosting", time.time() - start, str(exc))

    def _train_random_forest(
        self,
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train RandomForest with RandomizedSearchCV."""
        logger.info("=" * 50)
        logger.info("Training RandomForest with hyperparameter tuning")
        logger.info("=" * 50)

        start = time.time()
        try:
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model,
                RANDOM_FOREST_PARAMS,
                n_iter=15,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="f1_weighted",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train_smote, y_train_smote)
            model = search.best_estimator_
            best_params = search.best_params_
            logger.info("RandomForest best params: %s", best_params)

            skf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            cv_scores = cross_val_score(
                model, X_train_smote, y_train_smote, cv=skf, scoring="f1_weighted", n_jobs=-1
            )

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            metrics = _compute_metrics(y_test, y_pred, y_proba, self.label_mapping)
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_std"] = round(float(cv_scores.std()), 4)
            metrics["best_params"] = {str(k): str(v) for k, v in best_params.items()}
            metrics["training_time"] = round(time.time() - start, 2)

            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            metrics["feature_importances"] = [
                {
                    "name": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                    "importance": round(float(importances[i]), 6),
                }
                for i in sorted_idx
            ]

            model_path = os.path.join(self.models_dir, "RandomForest_model.joblib")
            joblib.dump(model, model_path)

            metrics["model_name"] = "RandomForest"
            metrics["model_path"] = model_path
            self.models["RandomForest"] = model

            logger.info(
                "RandomForest done — Acc: %.4f | F1W: %.4f | Time: %.2fs",
                metrics["accuracy"], metrics["f1_weighted"], metrics["training_time"],
            )
            return metrics

        except Exception as exc:
            logger.error("RandomForest training failed: %s", exc)
            traceback.print_exc()
            return self._empty_result("RandomForest", time.time() - start, str(exc))

    def _train_svm(
        self,
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train SVM with RandomizedSearchCV (RBF kernel)."""
        logger.info("=" * 50)
        logger.info("Training SVM with hyperparameter tuning")
        logger.info("=" * 50)

        start = time.time()
        try:
            base_model = SVC(probability=True, random_state=self.random_state)
            search = RandomizedSearchCV(
                base_model,
                SVM_PARAMS,
                n_iter=10,
                cv=StratifiedKFold(n_splits=min(self.cv_folds, 3), shuffle=True, random_state=self.random_state),
                scoring="f1_weighted",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train_smote, y_train_smote)
            model = search.best_estimator_
            best_params = search.best_params_
            logger.info("SVM best params: %s", best_params)

            skf = StratifiedKFold(
                n_splits=min(self.cv_folds, 3), shuffle=True, random_state=self.random_state
            )
            cv_scores = cross_val_score(
                model, X_train_smote, y_train_smote, cv=skf, scoring="f1_weighted", n_jobs=-1
            )

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            metrics = _compute_metrics(y_test, y_pred, y_proba, self.label_mapping)
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_std"] = round(float(cv_scores.std()), 4)
            metrics["best_params"] = {str(k): str(v) for k, v in best_params.items()}
            metrics["training_time"] = round(time.time() - start, 2)
            metrics["feature_importances"] = []  # SVM has no native feature importances

            model_path = os.path.join(self.models_dir, "SVM_model.joblib")
            joblib.dump(model, model_path)

            metrics["model_name"] = "SVM"
            metrics["model_path"] = model_path
            self.models["SVM"] = model

            logger.info(
                "SVM done — Acc: %.4f | F1W: %.4f | Time: %.2fs",
                metrics["accuracy"], metrics["f1_weighted"], metrics["training_time"],
            )
            return metrics

        except Exception as exc:
            logger.error("SVM training failed: %s", exc)
            traceback.print_exc()
            return self._empty_result("SVM", time.time() - start, str(exc))

    def _train_logistic_regression(
        self,
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train LogisticRegression with RandomizedSearchCV."""
        logger.info("=" * 50)
        logger.info("Training LogisticRegression with hyperparameter tuning")
        logger.info("=" * 50)

        start = time.time()
        try:
            base_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                n_jobs=-1,
            )
            search = RandomizedSearchCV(
                base_model,
                LOGISTIC_REGRESSION_PARAMS,
                n_iter=10,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring="f1_weighted",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train_smote, y_train_smote)
            model = search.best_estimator_
            best_params = search.best_params_
            logger.info("LogisticRegression best params: %s", best_params)

            skf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            cv_scores = cross_val_score(
                model, X_train_smote, y_train_smote, cv=skf, scoring="f1_weighted", n_jobs=-1
            )

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            metrics = _compute_metrics(y_test, y_pred, y_proba, self.label_mapping)
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_std"] = round(float(cv_scores.std()), 4)
            metrics["best_params"] = {str(k): str(v) for k, v in best_params.items()}
            metrics["training_time"] = round(time.time() - start, 2)

            # Coefficients as feature importances
            if hasattr(model, "coef_"):
                coef = np.abs(model.coef_).mean(axis=0)
                sorted_idx = np.argsort(coef)[::-1]
                metrics["feature_importances"] = [
                    {
                        "name": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                        "importance": round(float(coef[i]), 6),
                    }
                    for i in sorted_idx
                ]
            else:
                metrics["feature_importances"] = []

            model_path = os.path.join(self.models_dir, "LogisticRegression_model.joblib")
            joblib.dump(model, model_path)

            metrics["model_name"] = "LogisticRegression"
            metrics["model_path"] = model_path
            self.models["LogisticRegression"] = model

            logger.info(
                "LogisticRegression done — Acc: %.4f | F1W: %.4f | Time: %.2fs",
                metrics["accuracy"], metrics["f1_weighted"], metrics["training_time"],
            )
            return metrics

        except Exception as exc:
            logger.error("LogisticRegression training failed: %s", exc)
            traceback.print_exc()
            return self._empty_result("LogisticRegression", time.time() - start, str(exc))

    @staticmethod
    def _empty_result(model_name: str, training_time: float, error: str) -> Dict[str, Any]:
        """Return an empty result dict for a failed model."""
        return {
            "model_name": model_name,
            "model": None,
            "error": error,
            "training_time": round(training_time, 2),
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
            "classification_report": "",
            "class_labels": [],
            "feature_importances": [],
            "best_params": {},
        }

    # ───────────────────────────────────────
    # FULL PIPELINE
    # ───────────────────────────────────────

    def run(self, segment_mapping_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Execute the full Delhi training pipeline.

        Steps:
            1. Load segment mapping data
            2. Engineer features & assign risk classes
            3. Preprocess (split, scale, SMOTE)
            4. Train all 5 models with hyperparameter tuning
            5. Evaluate and save all artifacts

        Args:
            segment_mapping_path: Override path to segment mapping JSON.

        Returns:
            Dict of model_name → result dict with metrics and paths.
        """
        pipeline_start = time.time()
        logger.info("=" * 60)
        logger.info("DELHI ACCIDENT SEVERITY TRAINING PIPELINE")
        logger.info("=" * 60)

        # Step 1 & 2: Load and engineer features
        logger.info("[Step 1/5] Loading data and engineering features...")
        df = self.load_data(segment_mapping_path)

        # Step 3: Preprocess
        logger.info("[Step 2/5] Preprocessing (split, scale, SMOTE)...")
        (
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            X_train_smote, y_train_smote,
            _scaler, _le, _lm,
        ) = self.preprocess(df)

        # Step 4: Train all models
        logger.info("[Step 3/5] Training models...")
        trainer_args = (X_train_smote, y_train_smote, X_test_scaled, y_test)

        self.results = {}
        self.results["XGBoost"] = self._train_xgboost(*trainer_args)
        self.results["GradientBoosting"] = self._train_gradient_boosting(*trainer_args)
        self.results["RandomForest"] = self._train_random_forest(*trainer_args)
        self.results["SVM"] = self._train_svm(*trainer_args)
        self.results["LogisticRegression"] = self._train_logistic_regression(*trainer_args)

        # Step 5: Evaluate & summarize
        logger.info("[Step 4/5] Summarizing results...")
        successful = {k: v for k, v in self.results.items() if v.get("model") is not None}
        failed = {k: v for k, v in self.results.items() if v.get("model") is None}

        best_model = None
        best_f1 = 0.0
        if successful:
            best_name = max(successful, key=lambda k: successful[k].get("f1_weighted", 0))
            best_model = best_name
            best_f1 = successful[best_name]["f1_weighted"]

        # Training summary
        self.training_summary = {
            "dataset": "delhi",
            "pipeline": "delhi_trainer",
            "timestamp": datetime.now().isoformat(),
            "total_pipeline_time_sec": round(time.time() - pipeline_start, 2),
            "total_segments": len(df),
            "train_size_original": int(X_train_scaled.shape[0]),
            "train_size_after_smote": int(X_train_smote.shape[0]),
            "test_size": int(X_test_scaled.shape[0]),
            "n_features": int(X_train_scaled.shape[1]),
            "feature_names": self.feature_names,
            "label_mapping": self.label_mapping,
            "models_trained": list(successful.keys()),
            "models_failed": list(failed.keys()),
            "best_model": best_model,
            "best_f1_weighted": best_f1,
            "model_results": {
                name: {
                    "accuracy": res.get("accuracy", 0),
                    "f1_weighted": res.get("f1_weighted", 0),
                    "f1_macro": res.get("f1_macro", 0),
                    "roc_auc": res.get("roc_auc", 0),
                    "cohens_kappa": res.get("cohens_kappa", 0),
                    "mcc": res.get("mcc", 0),
                    "cv_mean": res.get("cv_mean", 0),
                    "cv_std": res.get("cv_std", 0),
                    "training_time": res.get("training_time", 0),
                    "best_params": res.get("best_params", {}),
                    "error": res.get("error"),
                }
                for name, res in self.results.items()
            },
        }

        # Save training summary
        summary_path = os.path.join(self.models_dir, "training_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(self.training_summary, fh, indent=2, default=str)
        logger.info("Training summary saved to %s", summary_path)

        # Print summary
        logger.info("[Step 5/5] Pipeline complete!")
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info("Models trained: %d/%d", len(successful), len(self.results))
        if best_model:
            logger.info("Best model: %s (F1-W: %.4f)", best_model, best_f1)
        if failed:
            logger.info("Failed models: %s", list(failed.keys()))
            for name, res in failed.items():
                logger.info("  %s: %s", name, res.get("error", "Unknown"))
        logger.info("Total pipeline time: %.2fs", time.time() - pipeline_start)
        logger.info("Artifacts saved to: %s", self.models_dir)

        return self.results


# ─────────────────────────────────────────────
# DELHI-SPECIFIC PREDICTOR CLASS
# ─────────────────────────────────────────────


class DelhiPredictor:
    """Delhi-specific predictor that loads trained models and makes predictions.

    Loads models, scaler, and label encoder from outputs/models/delhi/.

    Example::

        predictor = DelhiPredictor()
        predictor.load()
        result = predictor.predict({
            "total_accidents": 10,
            "fatal_count": 2,
            "grievous_count": 3,
            "minor_count": 5,
            "fatal_rate": 0.2,
            "accident_density": 15.5,
            "weighted_severity": 5.5,
            "length_m": 645.0,
            "num_years": 3,
            "year_span": 2,
            "day_accident_ratio": 0.6,
            "is_highway": 1,
            "is_urban": 0,
            "is_intersection": 0,
            "is_virtual": 0,
        })
    """

    def __init__(self, models_dir: Optional[str] = None) -> None:
        self.models_dir = models_dir or DELHI_MODELS_DIR
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = list(FEATURE_COLUMNS)
        self.label_mapping: Dict[str, str] = {}
        self.loaded: bool = False

    def load(self) -> bool:
        """Load all trained models and preprocessing artifacts.

        Returns:
            True if at least one model was loaded successfully.
        """
        logger.info("Loading Delhi predictor artifacts from %s", self.models_dir)

        # Scaler
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
        else:
            logger.warning("Scaler not found at %s", scaler_path)

        # Label encoder
        le_path = os.path.join(self.models_dir, "label_encoder.joblib")
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            logger.info("Label encoder loaded")
        else:
            logger.warning("Label encoder not found at %s", le_path)

        # Feature info
        fi_path = os.path.join(self.models_dir, "feature_info.json")
        if os.path.exists(fi_path):
            with open(fi_path, "r") as fh:
                fi = json.load(fh)
            self.feature_names = fi.get("feature_names", list(FEATURE_COLUMNS))
            self.label_mapping = fi.get("label_mapping", {})
            logger.info("Feature info loaded (%d features)", len(self.feature_names))
        else:
            logger.warning("Feature info not found at %s", fi_path)

        # Models
        model_names = ["XGBoost", "GradientBoosting", "RandomForest", "SVM", "LogisticRegression"]
        for name in model_names:
            model_path = os.path.join(self.models_dir, f"{name}_model.joblib")
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                    logger.info("%s model loaded from %s", name, model_path)
                except Exception as exc:
                    logger.warning("Failed to load %s model: %s", name, exc)
            else:
                logger.warning("%s model not found at %s", name, model_path)

        self.loaded = len(self.models) > 0
        logger.info(
            "%s: %d models loaded",
            "OK" if self.loaded else "ERROR", len(self.models),
        )
        return self.loaded

    def _prepare_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Convert input dict to feature vector matching training features.

        Args:
            input_data: Dict with feature names as keys.

        Returns:
            Scaled 2D feature array of shape (1, n_features).

        Raises:
            ValueError: If models/artifacts not loaded.
        """
        if not self.loaded:
            raise ValueError("Models not loaded. Call load() first.")

        feature_vector = np.zeros(len(self.feature_names), dtype=np.float64)
        for i, fname in enumerate(self.feature_names):
            if fname in input_data:
                val = input_data[fname]
                if isinstance(val, (int, float)):
                    feature_vector[i] = float(val)

        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector = feature_vector.reshape(1, -1)

        return feature_vector

    def predict(
        self, input_data: Dict[str, Any], model_name: str = "XGBoost"
    ) -> Dict[str, Any]:
        """Make a risk class prediction for a Delhi road segment.

        Args:
            input_data: Dict with feature values.
            model_name: Which model to use.

        Returns:
            Dict with prediction, confidence, probabilities, etc.
        """
        if not self.loaded:
            return {"error": "Models not loaded. Call load() first."}

        if model_name not in self.models:
            available = list(self.models.keys())
            if available:
                model_name = available[0]
                logger.info("Requested model not found. Using %s", model_name)
            else:
                return {"error": "No models available."}

        model = self.models[model_name]
        X = self._prepare_input(input_data)
        prediction_code = int(model.predict(X)[0])

        prediction_label = self.label_mapping.get(str(prediction_code), str(prediction_code))

        probabilities: Dict[str, float] = {}
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

        # Top risk factors from feature importances
        top_risk_factors = []
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:5]
            for idx in sorted_idx:
                if idx < len(self.feature_names):
                    fname = self.feature_names[idx]
                    val = input_data.get(fname, 0)
                    top_risk_factors.append({
                        "feature": fname,
                        "value": val,
                        "importance": round(float(importances[idx]), 6),
                    })

        return {
            "prediction": prediction_label,
            "prediction_code": prediction_code,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_risk_factors": top_risk_factors,
            "model_used": model_name,
        }

    def predict_batch(
        self, segments: List[Dict[str, Any]], model_name: str = "XGBoost"
    ) -> Dict[str, Any]:
        """Make predictions for a batch of Delhi road segments.

        Args:
            segments: List of feature dicts.
            model_name: Which model to use.

        Returns:
            Dict with total count, predictions list, and summary.
        """
        predictions = []
        for i, seg in enumerate(segments):
            result = self.predict(seg, model_name)
            result["segment_index"] = i
            predictions.append(result)

        summary: Dict[str, int] = {}
        for p in predictions:
            label = p.get("prediction", "Unknown")
            summary[label] = summary.get(label, 0) + 1

        return {
            "total_segments": len(predictions),
            "predictions": predictions,
            "summary": summary,
        }

    def get_available_models(self) -> List[str]:
        """Return list of loaded model names."""
        return list(self.models.keys())

    def get_feature_names(self) -> List[str]:
        """Return list of feature names used by models."""
        return self.feature_names

    def get_label_mapping(self) -> Dict[str, str]:
        """Return label mapping dict."""
        return self.label_mapping


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    """Run the Delhi training pipeline from command line."""
    trainer = DelhiTrainer()
    results = trainer.run()

    # Quick validation: try loading with predictor
    logger.info("\nValidating saved models with DelhiPredictor...")
    predictor = DelhiPredictor()
    if predictor.load():
        # Test prediction with sample data
        sample = {
            "total_accidents": 10,
            "fatal_count": 2,
            "grievous_count": 3,
            "minor_count": 5,
            "fatal_rate": 0.2,
            "accident_density": 15.5,
            "weighted_severity": 5.5,
            "length_m": 645.0,
            "num_years": 3,
            "year_span": 2,
            "day_accident_ratio": 0.6,
            "is_highway": 1,
            "is_urban": 0,
            "is_intersection": 0,
            "is_virtual": 0,
        }
        result = predictor.predict(sample)
        logger.info("Sample prediction: %s", result)
    else:
        logger.warning("Predictor validation failed — no models loaded.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
