"""
Standalone script to run the complete ML training pipeline.
Run this BEFORE starting the API server.

Usage: cd backend && python run_pipeline.py
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUTS_DIR, MODELS_DIR, PRIMARY_DATASET, SECONDARY_DATASET
from ml.data_loader import load_all_datasets, generate_eda_summary
from ml.preprocessor import preprocess_dataset
from ml.trainer import train_all_models
from ml.evaluator import evaluate_all_models, generate_evaluation_plots
from ml.shap_analyzer import run_shap_all_models


def generate_chart_data(datasets):
    """Generate chart-ready JSON data from datasets for frontend consumption."""
    import pandas as pd
    import numpy as np

    for key, ds in datasets.items():
        df = ds["dataframe"]
        target = ds["target_column"]
        roles = ds["roles"]

        dist = df[target].value_counts().to_dict()
        chart = {
            "chart_type": "donut",
            "title": "Accident Severity Distribution",
            "labels": [str(k) for k in dist.keys()],
            "values": [int(v) for v in dist.values()],
            "colors": ["#EF4444", "#F97316", "#F59E0B", "#10B981"][: len(dist)],
        }
        path = os.path.join(OUTPUTS_DIR, f"chart_data_class_distribution_{key}.json")
        with open(path, "w") as f:
            json.dump(chart, f, indent=2)
        print(f"  [SAVED] class_distribution chart data → {path}")

        time_cols = roles.get("time_columns", [])
        for tc in time_cols:
            if tc in df.columns:
                try:
                    time_series = pd.to_datetime(
                        df[tc].astype(str), format="%H:%M", errors="coerce"
                    )
                    if time_series.isna().all():
                        time_series = pd.to_datetime(
                            df[tc].astype(str), errors="coerce"
                        )
                    hours = time_series.dt.hour.dropna().astype(int)
                    hour_counts = hours.value_counts().sort_index()
                    chart = {
                        "chart_type": "area",
                        "title": "Accidents by Hour of Day",
                        "labels": list(range(24)),
                        "values": [int(hour_counts.get(h, 0)) for h in range(24)],
                    }
                    path = os.path.join(
                        OUTPUTS_DIR, f"chart_data_accidents_by_hour_{key}.json"
                    )
                    with open(path, "w") as f:
                        json.dump(chart, f, indent=2)
                    print(f"  [SAVED] accidents_by_hour chart data → {path}")
                    break
                except Exception as e:
                    print(f"  [WARNING] Could not generate time chart from '{tc}': {e}")

        day_col = None
        for col in df.columns:
            if "day" in col.lower():
                day_col = col
                break

        if day_col and day_col in df.columns:
            day_counts = df[day_col].value_counts().to_dict()
            chart = {
                "chart_type": "bar",
                "title": "Accidents by Day of Week",
                "labels": [str(k) for k in day_counts.keys()],
                "values": [int(v) for v in day_counts.values()],
            }
            path = os.path.join(OUTPUTS_DIR, f"chart_data_accidents_by_day_{key}.json")
            with open(path, "w") as f:
                json.dump(chart, f, indent=2)
            print(f"  [SAVED] accidents_by_day chart data → {path}")

        weather_cols = roles.get("weather_columns", [])
        if weather_cols:
            wc = weather_cols[0]
            if wc in df.columns:
                weather_severity = pd.crosstab(df[wc], df[target])
                datasets_chart = []
                severity_colors = ["#EF4444", "#F97316", "#F59E0B", "#10B981"]
                for i, sev_class in enumerate(weather_severity.columns):
                    datasets_chart.append(
                        {
                            "label": str(sev_class),
                            "data": weather_severity[sev_class].tolist(),
                            "color": severity_colors[i % len(severity_colors)],
                        }
                    )
                chart = {
                    "chart_type": "grouped_bar",
                    "title": "Weather Impact on Severity",
                    "labels": [str(x) for x in weather_severity.index.tolist()],
                    "datasets": datasets_chart,
                }
                path = os.path.join(
                    OUTPUTS_DIR, f"chart_data_accidents_by_weather_{key}.json"
                )
                with open(path, "w") as f:
                    json.dump(chart, f, indent=2)
                print(f"  [SAVED] accidents_by_weather chart data → {path}")

        vehicle_cols = roles.get("vehicle_columns", [])
        if vehicle_cols:
            vc = vehicle_cols[0]
            if vc in df.columns:
                veh_counts = df[vc].value_counts().head(10).to_dict()
                chart = {
                    "chart_type": "pie",
                    "title": "Vehicle Type Distribution",
                    "labels": [str(k) for k in veh_counts.keys()],
                    "values": [int(v) for v in veh_counts.values()],
                    "colors": [
                        "#2563EB",
                        "#EF4444",
                        "#10B981",
                        "#F59E0B",
                        "#8B5CF6",
                        "#F97316",
                        "#06B6D4",
                        "#EC4899",
                        "#84CC16",
                        "#A855F7",
                    ],
                }
                path = os.path.join(
                    OUTPUTS_DIR, f"chart_data_accidents_by_vehicle_{key}.json"
                )
                with open(path, "w") as f:
                    json.dump(chart, f, indent=2)
                print(f"  [SAVED] accidents_by_vehicle chart data → {path}")

        cause_cols = roles.get("cause_columns", [])
        if cause_cols:
            cc = cause_cols[0]
            if cc in df.columns:
                cause_severity = pd.crosstab(df[cc], df[target])
                cause_severity["Total"] = cause_severity.sum(axis=1)
                cause_severity = cause_severity.sort_values(
                    "Total", ascending=False
                ).head(10)
                cause_severity = cause_severity.drop("Total", axis=1)

                datasets_chart = []
                severity_colors = ["#EF4444", "#F97316", "#F59E0B", "#10B981"]
                for i, sev_class in enumerate(cause_severity.columns):
                    datasets_chart.append(
                        {
                            "label": str(sev_class),
                            "data": cause_severity[sev_class].tolist(),
                            "color": severity_colors[i % len(severity_colors)],
                        }
                    )

                chart = {
                    "chart_type": "horizontal_bar",
                    "title": "Top Accident Causes by Severity",
                    "labels": [str(x) for x in cause_severity.index.tolist()],
                    "datasets": datasets_chart,
                }
                path = os.path.join(
                    OUTPUTS_DIR, f"chart_data_severity_by_cause_{key}.json"
                )
                with open(path, "w") as f:
                    json.dump(chart, f, indent=2)
                print(f"  [SAVED] severity_by_cause chart data → {path}")

        date_cols = roles.get("date_columns", [])
        if date_cols:
            dc = date_cols[0]
            if dc in df.columns:
                try:
                    parsed = pd.to_datetime(df[dc], errors="coerce", dayfirst=True)
                    df_temp = df.copy()
                    df_temp["_parsed_date"] = parsed
                    df_temp["_month"] = parsed.dt.to_period("M").astype(str)
                    valid = df_temp.dropna(subset=["_parsed_date"])

                    if len(valid) > 0:
                        monthly = pd.crosstab(valid["_month"], valid[target])
                        datasets_chart = []
                        severity_colors = ["#EF4444", "#F97316", "#F59E0B", "#10B981"]
                        for i, sev_class in enumerate(monthly.columns):
                            datasets_chart.append(
                                {
                                    "label": str(sev_class),
                                    "data": monthly[sev_class].tolist(),
                                    "color": severity_colors[i % len(severity_colors)],
                                }
                            )

                        chart = {
                            "chart_type": "line",
                            "title": "Monthly Accident Trend",
                            "labels": monthly.index.tolist(),
                            "datasets": datasets_chart,
                        }
                        path = os.path.join(
                            OUTPUTS_DIR, f"chart_data_monthly_trend_{key}.json"
                        )
                        with open(path, "w") as f:
                            json.dump(chart, f, indent=2)
                        print(f"  [SAVED] monthly_trend chart data → {path}")
                except Exception as e:
                    print(f"  [WARNING] Could not generate monthly trend: {e}")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr().round(3)
            chart = {
                "chart_type": "heatmap",
                "title": "Feature Correlation Matrix",
                "labels": corr.columns.tolist(),
                "values": corr.values.tolist(),
            }
            path = os.path.join(
                OUTPUTS_DIR, f"chart_data_correlation_matrix_{key}.json"
            )
            with open(path, "w") as f:
                json.dump(chart, f, indent=2)
            print(f"  [SAVED] correlation_matrix chart data → {path}")

    filter_options = {}
    for key, ds in datasets.items():
        df = ds["dataframe"]
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) <= 50:
                col_clean = col.replace(" ", "_")
                filter_options[col_clean] = [str(v) for v in unique_vals[:50]]

    filter_path = os.path.join(OUTPUTS_DIR, "filter_options.json")
    with open(filter_path, "w") as f:
        json.dump(filter_options, f, indent=2)
    print(f"  [SAVED] filter_options → {filter_path}")


def run_full_pipeline():
    """Execute the complete ML pipeline end-to-end."""
    start_total = time.time()

    print("\n" + "=" * 60)
    print("  INDIAN ROAD ACCIDENT SEVERITY PREDICTION")
    print("  Full ML Training Pipeline")
    print("=" * 60)

    print("\n[STEP 1/6] Loading Datasets...")
    datasets = load_all_datasets()
    if not datasets:
        print("\n  [FATAL] No datasets available. Cannot proceed.")
        print(f"  Place your CSV files in: {DATA_DIR}")
        print(f"  Expected files: {PRIMARY_DATASET}, {SECONDARY_DATASET}")
        sys.exit(1)

    print("\n[STEP 2/6] Generating EDA Summary & Chart Data...")
    eda_summary = generate_eda_summary(datasets)
    generate_chart_data(datasets)

    all_pipeline_results = {}

    for key, ds in datasets.items():
        print(f"\n{'#'*60}")
        print(f"  PIPELINE FOR: {ds['name']}")
        print(f"{'#'*60}")

        df = ds["dataframe"]
        target = ds["target_column"]
        roles = ds["roles"]

        print(f"\n[STEP 3/6] Preprocessing: {key}...")
        X, y, feature_names, le, label_mapping = preprocess_dataset(
            df, target, roles, key
        )

        print(f"\n[STEP 4/6] Training 5 Models: {key}...")
        results, X_train, X_test, y_train, y_test, X_smote, y_smote = train_all_models(
            X, y, feature_names, label_mapping, key
        )

        print(f"\n[STEP 5/6] Evaluating Models: {key}...")
        all_metrics = evaluate_all_models(results, X_test, y_test, label_mapping, key)
        generate_evaluation_plots(
            all_metrics, results, X_test, y_test, label_mapping, key
        )

        print(f"\n[STEP 6/6] SHAP Analysis (All 5 Models): {key}...")
        shap_results = run_shap_all_models(
            results, X_test, feature_names, label_mapping, key
        )

        all_pipeline_results[key] = {
            "models_trained": len(
                [r for r in results.values() if r.get("model") is not None]
            ),
            "best_model": (
                max(all_metrics, key=lambda x: x["f1_weighted"])["model_name"]
                if all_metrics
                else None
            ),
            "best_f1": (
                max(all_metrics, key=lambda x: x["f1_weighted"])["f1_weighted"]
                if all_metrics
                else 0
            ),
            "metrics": [
                {
                    k: v
                    for k, v in m.items()
                    if k
                    not in [
                        "confusion_matrix",
                        "normalized_confusion_matrix",
                        "roc_data",
                    ]
                }
                for m in all_metrics
            ],
        }

    total_time = time.time() - start_total

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    for key, res in all_pipeline_results.items():
        print(f"\n  Dataset: {key}")
        print(f"    Models trained: {res['models_trained']}")
        print(f"    Best model: {res['best_model']}")
        print(f"    Best F1 (weighted): {res['best_f1']:.4f}")

    pipeline_summary_path = os.path.join(OUTPUTS_DIR, "pipeline_summary.json")
    with open(pipeline_summary_path, "w") as f:
        json.dump(
            {
                "total_time_seconds": round(total_time, 2),
                "datasets_processed": list(all_pipeline_results.keys()),
                "results": all_pipeline_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  [SAVED] Pipeline summary → {pipeline_summary_path}")

    print(f"\n  Next step: Start API server with:")
    print(f"    cd backend && python main.py")
    print(f"  Then test with:")
    print(f"    curl http://localhost:8000/api/health")

    return all_pipeline_results


if __name__ == "__main__":
    run_full_pipeline()
