from ml.models.random_forest import build_random_forest, train_random_forest
from ml.models.xgboost_model import build_xgboost, train_xgboost
from ml.models.gradient_boosting import build_gradient_boosting, train_gradient_boosting
from ml.models.svm_model import build_svm, train_svm
from ml.models.logistic_regression import (
    build_logistic_regression,
    train_logistic_regression,
)

MODEL_BUILDERS = {
    "RandomForest": build_random_forest,
    "XGBoost": build_xgboost,
    "GradientBoosting": build_gradient_boosting,
    "SVM": build_svm,
    "LogisticRegression": build_logistic_regression,
}

MODEL_TRAINERS = {
    "RandomForest": train_random_forest,
    "XGBoost": train_xgboost,
    "GradientBoosting": train_gradient_boosting,
    "SVM": train_svm,
    "LogisticRegression": train_logistic_regression,
}

__all__ = [
    "MODEL_BUILDERS",
    "MODEL_TRAINERS",
    "build_random_forest",
    "train_random_forest",
    "build_xgboost",
    "train_xgboost",
    "build_gradient_boosting",
    "train_gradient_boosting",
    "build_svm",
    "train_svm",
    "build_logistic_regression",
    "train_logistic_regression",
]
