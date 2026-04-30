import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import OUTPUTS_DIR, MODELS_DIR, RANDOM_STATE


def extract_time_features(df, roles):
    """Extract temporal features from time and date columns."""
    time_cols = roles.get("time_columns", [])
    date_cols = roles.get("date_columns", [])

    for col in time_cols:
        if col in df.columns:
            try:
                time_series = df[col].astype(str).str.strip()
                parsed_time = pd.to_datetime(
                    time_series, format="%H:%M", errors="coerce"
                )
                if parsed_time.isna().all():
                    parsed_time = pd.to_datetime(time_series, errors="coerce")

                df["Hour"] = parsed_time.dt.hour.fillna(12).astype(int)
                df["Time_Period"] = pd.cut(
                    df["Hour"],
                    bins=[-1, 6, 12, 17, 21, 24],
                    labels=["Night", "Morning", "Afternoon", "Evening", "Night_Late"],
                ).astype(str)
                df["Time_Period"] = df["Time_Period"].replace("Night_Late", "Night")
                df["Is_Night"] = ((df["Hour"] >= 21) | (df["Hour"] < 6)).astype(int)
                print(f"  [INFO] Extracted Hour, Time_Period, Is_Night from '{col}'")
                break
            except Exception as e:
                print(f"  [WARNING] Could not parse time column '{col}': {e}")

    for col in date_cols:
        if col in df.columns:
            try:
                parsed_date = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                if parsed_date.notna().sum() > len(df) * 0.3:
                    df["Month"] = parsed_date.dt.month.fillna(1).astype(int)
                    df["Year"] = parsed_date.dt.year.fillna(2020).astype(int)
                    df["Quarter"] = parsed_date.dt.quarter.fillna(1).astype(int)
                    df["DayOfMonth"] = parsed_date.dt.day.fillna(15).astype(int)
                    print(
                        f"  [INFO] Extracted Month, Year, Quarter, DayOfMonth from '{col}'"
                    )
                    break
            except Exception as e:
                print(f"  [WARNING] Could not parse date column '{col}': {e}")

    day_col = None
    for col in df.columns:
        if "day" in col.lower() and "week" in col.lower():
            day_col = col
            break
        elif col.lower() in ["day", "day_of_week", "dayofweek"]:
            day_col = col
            break

    if day_col and day_col in df.columns:
        day_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        if df[day_col].dtype == "object":
            df["DayOfWeek_Num"] = (
                df[day_col].str.lower().str.strip().map(day_map).fillna(0).astype(int)
            )
        else:
            df["DayOfWeek_Num"] = df[day_col].fillna(0).astype(int)

        df["Is_Weekend"] = (df["DayOfWeek_Num"] >= 5).astype(int)
        print(f"  [INFO] Extracted DayOfWeek_Num, Is_Weekend from '{day_col}'")

    return df


def create_interaction_features(df, roles):
    """Create interaction features from weather and road columns."""
    weather_cols = roles.get("weather_columns", [])
    road_cols = roles.get("road_columns", [])

    weather_col = weather_cols[0] if weather_cols else None
    road_col = road_cols[0] if road_cols else None

    if (
        weather_col
        and road_col
        and weather_col in df.columns
        and road_col in df.columns
    ):
        df["Weather_Road_Interaction"] = (
            df[weather_col].astype(str).str.strip()
            + "_"
            + df[road_col].astype(str).str.strip()
        )
        print(
            f"  [INFO] Created Weather_Road_Interaction from '{weather_col}' + '{road_col}'"
        )

    if "Time_Period" in df.columns:
        location_cols = roles.get("location_columns", [])
        loc_col = location_cols[0] if location_cols else None
        if loc_col and loc_col in df.columns:
            df["Time_Location_Interaction"] = (
                df["Time_Period"].astype(str)
                + "_"
                + df[loc_col].astype(str).str.strip()
            )
            print(f"  [INFO] Created Time_Location_Interaction")

    return df


def handle_missing_values(df):
    """Impute missing values: median for numeric, mode for categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("Unknown")

    remaining_missing = df.isnull().sum().sum()
    print(f"  [INFO] Missing values after imputation: {remaining_missing}")
    return df


def encode_features(df, target_col):
    """Encode categorical features: one-hot for low cardinality, frequency for high."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    low_cardinality = []
    high_cardinality = []

    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique <= 15:
            low_cardinality.append(col)
        else:
            high_cardinality.append(col)

    print(f"  [INFO] Low cardinality columns (one-hot, <=15 unique): {low_cardinality}")
    print(
        f"  [INFO] High cardinality columns (frequency encode, >15 unique): {high_cardinality}"
    )

    for col in high_cardinality:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[col + "_freq"] = df[col].map(freq_map).fillna(0.0)
        df.drop(col, axis=1, inplace=True)

    if low_cardinality:
        df = pd.get_dummies(df, columns=low_cardinality, drop_first=True, dtype=int)
        print(f"  [INFO] One-hot encoded {len(low_cardinality)} columns")

    return df


def encode_target(df, target_col):
    """Encode the target variable to numeric labels."""
    le = LabelEncoder()
    original_classes = df[target_col].unique().tolist()

    df[target_col] = le.fit_transform(df[target_col].astype(str))

    label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print(f"  [INFO] Target encoding: {label_mapping}")

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    print(f"  [SAVED] Label encoder → {MODELS_DIR}/label_encoder.joblib")

    return df, le, label_mapping


def scale_features(X_train, X_test, feature_names):
    """Standard scale numeric features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print(f"  [SAVED] Scaler → {MODELS_DIR}/scaler.joblib")

    return X_train_scaled, X_test_scaled, scaler


def preprocess_dataset(df, target_col, roles, dataset_key="primary"):
    """Full preprocessing pipeline for a single dataset."""
    print(f"\n{'='*60}")
    print(f"  Preprocessing: {dataset_key}")
    print(f"{'='*60}")
    print(f"  Initial shape: {df.shape}")

    df = df.copy()

    leak_cols = roles.get("leak_columns", [])
    if leak_cols:
        existing_leak = [c for c in leak_cols if c in df.columns]
        if existing_leak:
            df.drop(columns=existing_leak, inplace=True)
            print(f"  [INFO] Dropped leak columns: {existing_leak}")

    cols_to_drop = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
        elif df[col].nunique() == len(df):
            if df[col].dtype == "object":
                cols_to_drop.append(col)

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  [INFO] Dropped constant/unique-ID columns: {cols_to_drop}")

    date_cols_to_drop = []
    for col in roles.get("date_columns", []):
        if col in df.columns:
            date_cols_to_drop.append(col)

    time_cols_to_drop = []
    for col in roles.get("time_columns", []):
        if col in df.columns:
            time_cols_to_drop.append(col)

    df = extract_time_features(df, roles)
    df = create_interaction_features(df, roles)

    for col in date_cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"  [INFO] Dropped raw date column: {col}")

    for col in time_cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"  [INFO] Dropped raw time column: {col}")

    df = handle_missing_values(df)

    df, label_encoder, label_mapping = encode_target(df, target_col)

    df = encode_features(df, target_col)

    remaining_object_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if target_col in remaining_object_cols:
        remaining_object_cols.remove(target_col)
    if remaining_object_cols:
        df.drop(columns=remaining_object_cols, inplace=True)
        print(
            f"  [INFO] Dropped remaining non-numeric columns: {remaining_object_cols}"
        )

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values
    y = df[target_col].values

    print(f"\n  Final feature matrix shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(
        f"  Feature names ({len(feature_names)}): {feature_names[:20]}{'...' if len(feature_names) > 20 else ''}"
    )
    print(
        f"  Class distribution after encoding: {pd.Series(y).value_counts().to_dict()}"
    )

    feature_info = {
        "feature_names": feature_names,
        "label_mapping": {str(k): str(v) for k, v in label_mapping.items()},
        "n_features": len(feature_names),
        "n_classes": len(label_mapping),
    }

    feature_info_path = os.path.join(MODELS_DIR, f"feature_info_{dataset_key}.json")
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"  [SAVED] Feature info → {feature_info_path}")

    return X, y, feature_names, label_encoder, label_mapping
