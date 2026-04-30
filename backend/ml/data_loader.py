import os
import pandas as pd
import numpy as np
import json
from config import (
    DATA_DIR,
    PRIMARY_PATH,
    SECONDARY_PATH,
    PRIMARY_DATASET,
    SECONDARY_DATASET,
    OUTPUTS_DIR,
)


def find_column_by_keywords(columns, keywords):
    """Find a column name that contains any of the given keywords (case-insensitive)."""
    for col in columns:
        col_lower = col.lower().strip()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return col
    return None


def find_columns_by_keywords(columns, keywords):
    """Find all column names that contain any of the given keywords."""
    matched = []
    for col in columns:
        col_lower = col.lower().strip()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                matched.append(col)
                break
    return matched


def detect_target_column(df):
    """Dynamically detect the severity/target column."""
    severity_keywords = [
        "severity",
        "accident_severity",
        "injury_severity",
        "crash_severity",
    ]
    target_col = find_column_by_keywords(df.columns, severity_keywords)

    if target_col is not None:
        print(f"  [INFO] Target column detected: '{target_col}'")
        return target_col, False

    killed_col = find_column_by_keywords(df.columns, ["killed", "fatal", "death"])
    grievous_col = find_column_by_keywords(
        df.columns, ["grievous", "serious", "severe"]
    )
    minor_col = find_column_by_keywords(df.columns, ["minor", "slight"])
    no_injury_col = find_column_by_keywords(
        df.columns, ["non_injury", "no_injury", "none", "no injury"]
    )

    if killed_col or grievous_col or minor_col:
        print("  [INFO] No severity column found. Deriving from casualty columns...")
        df["Severity"] = "No Injury"

        if minor_col:
            mask = df[minor_col].fillna(0).astype(float) > 0
            df.loc[mask, "Severity"] = "Minor"
        if grievous_col:
            mask = df[grievous_col].fillna(0).astype(float) > 0
            df.loc[mask, "Severity"] = "Grievous"
        if killed_col:
            mask = df[killed_col].fillna(0).astype(float) > 0
            df.loc[mask, "Severity"] = "Fatal"

        leak_cols = [
            c
            for c in [killed_col, grievous_col, minor_col, no_injury_col]
            if c is not None
        ]
        print(
            f"  [INFO] Derived severity column created. Leak columns to drop: {leak_cols}"
        )
        return "Severity", True, leak_cols

    print("  [WARNING] Could not detect target column. Will attempt manual inspection.")
    return None, False, []


def detect_column_roles(df):
    """Detect semantic roles of columns using keyword matching."""
    roles = {
        "time_columns": find_columns_by_keywords(df.columns, ["time", "hour"]),
        "date_columns": find_columns_by_keywords(
            df.columns, ["date", "day", "month", "year"]
        ),
        "weather_columns": find_columns_by_keywords(df.columns, ["weather", "climate"]),
        "road_columns": find_columns_by_keywords(
            df.columns, ["road", "surface", "pavement"]
        ),
        "vehicle_columns": find_columns_by_keywords(
            df.columns, ["vehicle", "car", "truck", "bus"]
        ),
        "location_columns": find_columns_by_keywords(
            df.columns, ["location", "area", "zone", "district", "state"]
        ),
        "cause_columns": find_columns_by_keywords(
            df.columns, ["cause", "reason", "factor"]
        ),
    }
    return roles


def load_primary_dataset():
    """Load and inspect the primary dataset (ETP_4_New_Data_Accidents.csv)."""
    if not os.path.exists(PRIMARY_PATH):
        print(f"  [ERROR] Primary dataset not found at: {PRIMARY_PATH}")
        print(f"  [INFO] Please download from Zenodo and place as: {PRIMARY_PATH}")
        return None, None, None

    print(f"\n{'='*60}")
    print(f"  Loading Primary Dataset: {PRIMARY_DATASET}")
    print(f"{'='*60}")

    df = pd.read_csv(PRIMARY_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")
    print(f"\n  First 3 rows:\n{df.head(3)}")
    print(f"\n  Missing values:\n{df.isnull().sum()}")

    result = detect_target_column(df)
    if result[0] is None:
        print("  [ERROR] Could not detect target column in primary dataset.")
        return df, None, detect_column_roles(df)

    if len(result) == 3:
        target_col, derived, leak_cols = result
    else:
        target_col, derived = result
        leak_cols = []

    print(f"\n  Target column: '{target_col}'")
    print(f"  Class distribution:\n{df[target_col].value_counts()}")

    roles = detect_column_roles(df)
    roles["target"] = target_col
    roles["derived"] = derived
    roles["leak_columns"] = leak_cols if derived else []

    return df, target_col, roles


def load_secondary_dataset():
    """Load and inspect the secondary dataset (Road.csv)."""
    if not os.path.exists(SECONDARY_PATH):
        print(f"  [ERROR] Secondary dataset not found at: {SECONDARY_PATH}")
        print(f"  [INFO] Please download from Kaggle and place as: {SECONDARY_PATH}")
        return None, None, None

    print(f"\n{'='*60}")
    print(f"  Loading Secondary Dataset: {SECONDARY_DATASET}")
    print(f"{'='*60}")

    df = pd.read_csv(SECONDARY_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")
    print(f"\n  First 3 rows:\n{df.head(3)}")
    print(f"\n  Missing values:\n{df.isnull().sum()}")

    result = detect_target_column(df)
    if result[0] is None:
        print("  [ERROR] Could not detect target column in secondary dataset.")
        return df, None, detect_column_roles(df)

    if len(result) == 3:
        target_col, derived, leak_cols = result
    else:
        target_col, derived = result
        leak_cols = []

    print(f"\n  Target column: '{target_col}'")
    print(f"  Class distribution:\n{df[target_col].value_counts()}")

    roles = detect_column_roles(df)
    roles["target"] = target_col
    roles["derived"] = derived
    roles["leak_columns"] = leak_cols if derived else []

    return df, target_col, roles


def load_all_datasets():
    """Load all available datasets and return a dict of results."""
    datasets = {}

    df1, target1, roles1 = load_primary_dataset()
    if df1 is not None and target1 is not None:
        datasets["primary"] = {
            "name": "NHAI Multi-Corridor (ETP_4_New_Data_Accidents)",
            "filename": PRIMARY_DATASET,
            "dataframe": df1,
            "target_column": target1,
            "roles": roles1,
            "records": len(df1),
            "features": len(df1.columns),
            "severity_classes": df1[target1].nunique(),
            "class_distribution": df1[target1].value_counts().to_dict(),
        }

    df2, target2, roles2 = load_secondary_dataset()
    if df2 is not None and target2 is not None:
        datasets["secondary"] = {
            "name": "Kaggle India Severity (Road)",
            "filename": SECONDARY_DATASET,
            "dataframe": df2,
            "target_column": target2,
            "roles": roles2,
            "records": len(df2),
            "features": len(df2.columns),
            "severity_classes": df2[target2].nunique(),
            "class_distribution": df2[target2].value_counts().to_dict(),
        }

    if not datasets:
        print("\n  [CRITICAL] No datasets loaded! Place CSV files in ./backend/data/")
    else:
        print(f"\n  [SUCCESS] {len(datasets)} dataset(s) loaded successfully.")

    return datasets


def generate_eda_summary(datasets):
    """Generate and save EDA summary JSON for all loaded datasets."""
    summary = {}
    for key, ds in datasets.items():
        df = ds["dataframe"]
        target = ds["target_column"]

        missing = df.isnull().sum().to_dict()
        missing_pct = {k: round(v / len(df) * 100, 2) for k, v in missing.items()}

        ds_summary = {
            "name": ds["name"],
            "filename": ds["filename"],
            "total_records": ds["records"],
            "total_features": ds["features"],
            "severity_classes": ds["severity_classes"],
            "class_distribution": {
                str(k): int(v) for k, v in ds["class_distribution"].items()
            },
            "missing_values": {str(k): int(v) for k, v in missing.items()},
            "missing_pct": {str(k): float(v) for k, v in missing_pct.items()},
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(
                df.select_dtypes(include=["object", "category"]).columns
            ),
        }

        date_cols = ds["roles"].get("date_columns", [])
        if date_cols:
            for dc in date_cols:
                try:
                    parsed = pd.to_datetime(df[dc], errors="coerce")
                    valid = parsed.dropna()
                    if len(valid) > 0:
                        ds_summary["date_range_start"] = str(valid.min().date())
                        ds_summary["date_range_end"] = str(valid.max().date())
                        break
                except Exception:
                    pass

        summary[key] = ds_summary

    output_path = os.path.join(OUTPUTS_DIR, "eda_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [SAVED] EDA summary → {output_path}")

    return summary


if __name__ == "__main__":
    datasets = load_all_datasets()
    if datasets:
        summary = generate_eda_summary(datasets)
        for key, s in summary.items():
            print(f"\n  Dataset: {s['name']}")
            print(f"  Records: {s['total_records']}, Features: {s['total_features']}")
            print(f"  Classes: {s['class_distribution']}")
