import numpy as np
import json
import os
import base64
import io
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from config import SHAP_DIR, MODELS_DIR, PLOTS_DIR, OUTPUTS_DIR


def compute_shap_for_model(model, model_name, X_test, feature_names, label_mapping, dataset_key="primary"):
    """Compute SHAP values for a single model and save results."""
    print(f"\n  {'─'*40}")
    print(f"  SHAP Analysis: {model_name}")
    print(f"  {'─'*40}")

    shap_result = {
        "model_name": model_name,
        "feature_importance": [],
        "summary_plot_base64": "",
        "bar_plot_base64": ""
    }

    try:
        max_samples = min(500, X_test.shape[0])
        X_sample = X_test[:max_samples]

        explainer = None
        shap_values = None

        # Determine which explainer to use based on model type
        # TreeExplainer works for RF and XGBoost multiclass
        # GradientBoosting multiclass needs KernelExplainer
        # SVM and LR always need KernelExplainer

        tree_models_safe = ["RandomForest", "XGBoost"]
        kernel_models = ["SVM", "LogisticRegression", "GradientBoosting"]

        if model_name in tree_models_safe:
            try:
                print(f"  Using TreeExplainer for {model_name}...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                print(f"  TreeExplainer succeeded.")
            except Exception as tree_err:
                print(f"  TreeExplainer failed: {tree_err}")
                print(f"  Falling back to KernelExplainer...")
                explainer = None

        if explainer is None or shap_values is None:
            # Use KernelExplainer as fallback for all other models
            # Use smaller background sample for speed
            bg_size = min(50, X_test.shape[0])
            sample_size = min(100, X_sample.shape[0])
            X_sample = X_sample[:sample_size]

            print(f"  Using KernelExplainer for {model_name}...")
            print(f"  Background samples: {bg_size}, Explain samples: {sample_size}")
            print(f"  This may take several minutes...")

            background = shap.sample(X_test, bg_size)

            if hasattr(model, "predict_proba"):
                explainer = shap.KernelExplainer(model.predict_proba, background)
            else:
                explainer = shap.KernelExplainer(model.predict, background)

            shap_values = explainer.shap_values(X_sample, nsamples=50)
            print(f"  KernelExplainer succeeded.")

        # Calculate mean absolute SHAP values across all classes
        if isinstance(shap_values, list):
            mean_abs_shap = np.zeros(X_sample.shape[1])
            for sv in shap_values:
                mean_abs_shap += np.abs(sv).mean(axis=0)
            mean_abs_shap /= len(shap_values)
        else:
            if shap_values.ndim == 3:
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

        top_n = min(15, len(mean_abs_shap))
        sorted_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

        feature_importance = []
        for idx in sorted_indices:
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            feature_importance.append({
                "name": fname,
                "importance": round(float(mean_abs_shap[idx]), 6)
            })

        shap_result["feature_importance"] = feature_importance

        # Save feature importance JSON
        fi_path = os.path.join(SHAP_DIR, f"shap_feature_importance_{model_name}_{dataset_key}.json")
        with open(fi_path, "w") as f:
            json.dump(feature_importance, f, indent=2)
        print(f"  [SAVED] SHAP feature importance -> {fi_path}")

        # Generate summary plot
        try:
            plt.figure(figsize=(10, 8))
            plt.style.use("dark_background")
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                  max_display=top_n, show=False, plot_type="bar")
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                  max_display=top_n, show=False)
            plt.tight_layout()
            summary_path = os.path.join(SHAP_DIR, f"shap_summary_{model_name}_{dataset_key}.png")
            plt.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor="#1B1B2F")
            plt.close("all")

            # Also save as base64
            buf = io.BytesIO()
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                  max_display=top_n, show=False, plot_type="bar")
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                  max_display=top_n, show=False)
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1B1B2F")
            plt.close("all")
            buf.seek(0)
            summary_b64 = base64.b64encode(buf.read()).decode("utf-8")
            shap_result["summary_plot_base64"] = f"data:image/png;base64,{summary_b64}"
            print(f"  [SAVED] SHAP summary plot -> {summary_path}")
        except Exception as plot_err:
            print(f"  [WARNING] Could not generate summary plot: {plot_err}")

        # Generate bar plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use("dark_background")
            names_top = [fi["name"] for fi in feature_importance]
            values_top = [fi["importance"] for fi in feature_importance]
            ax.barh(range(len(names_top)), values_top[::-1], color="#2563EB", alpha=0.8)
            ax.set_yticks(range(len(names_top)))
            ax.set_yticklabels(names_top[::-1], fontsize=8)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP Feature Importance — {model_name}")
            plt.tight_layout()
            bar_path = os.path.join(SHAP_DIR, f"shap_bar_{model_name}_{dataset_key}.png")
            plt.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor="#1B1B2F")

            buf2 = io.BytesIO()
            plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight", facecolor="#1B1B2F")
            plt.close("all")
            buf2.seek(0)
            bar_b64 = base64.b64encode(buf2.read()).decode("utf-8")
            shap_result["bar_plot_base64"] = f"data:image/png;base64,{bar_b64}"
            print(f"  [SAVED] SHAP bar plot -> {bar_path}")
        except Exception as bar_err:
            print(f"  [WARNING] Could not generate bar plot: {bar_err}")

        # Save SHAP values array
        try:
            shap_values_path = os.path.join(SHAP_DIR, f"shap_values_{model_name}_{dataset_key}.npy")
            if isinstance(shap_values, list):
                np.save(shap_values_path, np.array(shap_values, dtype=object), allow_pickle=True)
            else:
                np.save(shap_values_path, shap_values)
            print(f"  [SAVED] SHAP values array -> {shap_values_path}")
        except Exception as save_err:
            print(f"  [WARNING] Could not save SHAP values: {save_err}")

        print(f"  [DONE] SHAP analysis for {model_name} complete.")

    except Exception as e:
        print(f"  [ERROR] SHAP analysis failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()

    return shap_result


def run_shap_all_models(results, X_test, feature_names, label_mapping, dataset_key="primary"):
    """Run SHAP analysis for ALL trained models."""
    print(f"\n{'='*60}")
    print(f"  SHAP Analysis — All Models: {dataset_key}")
    print(f"{'='*60}")

    all_shap_results = {}

    for model_name, result in results.items():
        model = result.get("model")
        if model is None:
            print(f"  [SKIP] {model_name} — not trained")
            continue

        shap_result = compute_shap_for_model(
            model, model_name, X_test, feature_names, label_mapping, dataset_key
        )
        all_shap_results[model_name] = shap_result

    # Save all SHAP results summary
    all_shap_path = os.path.join(SHAP_DIR, f"all_shap_results_{dataset_key}.json")
    serializable = {}
    for k, v in all_shap_results.items():
        serializable[k] = {
            "model_name": v["model_name"],
            "feature_importance": v["feature_importance"]
        }
    with open(all_shap_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  [SAVED] All SHAP results -> {all_shap_path}")

    # Save best model SHAP as the default
    valid_results = {k: v for k, v in all_shap_results.items() if v["feature_importance"]}
    if valid_results:
        best_model = max(valid_results.items(),
                         key=lambda x: x[1]["feature_importance"][0]["importance"]
                         if x[1]["feature_importance"] else 0)
        combined_path = os.path.join(OUTPUTS_DIR, "shap_feature_importance.json")
        with open(combined_path, "w") as f:
            json.dump({
                "best_model": best_model[0],
                "features": best_model[1]["feature_importance"]
            }, f, indent=2)
        print(f"  [SAVED] Best model SHAP -> {combined_path}")
    else:
        print(f"  [WARNING] No valid SHAP results to save as default")

    return all_shap_results
