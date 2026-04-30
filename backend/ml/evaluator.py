import numpy as np
import pandas as pd
import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from config import OUTPUTS_DIR, PLOTS_DIR


def evaluate_all_models(results, X_test, y_test, label_mapping, dataset_key="primary"):
    """
    Collect metrics from all trained models and generate comparison table.
    Each model's train function already computed metrics, so we just aggregate.
    """
    print(f"\n{'='*60}")
    print(f"  Collecting Evaluation Results: {dataset_key}")
    print(f"{'='*60}")

    all_metrics = []

    for model_name, result in results.items():
        if result.get("model") is None:
            print(f"  [SKIP] {model_name} — training failed")
            continue

        metrics = {
            "model_name": result["model_name"],
            "accuracy": result["accuracy"],
            "precision_weighted": result["precision_weighted"],
            "recall_weighted": result["recall_weighted"],
            "f1_weighted": result["f1_weighted"],
            "f1_macro": result["f1_macro"],
            "roc_auc": result["roc_auc"],
            "cohens_kappa": result["cohens_kappa"],
            "mcc": result["mcc"],
            "log_loss": result["log_loss"],
            "cv_mean": result["cv_mean"],
            "cv_std": result["cv_std"],
            "training_time": result["training_time"],
            "confusion_matrix": result["confusion_matrix"],
            "normalized_confusion_matrix": result["normalized_confusion_matrix"],
            "roc_data": result["roc_data"],
            "class_labels": result["class_labels"],
        }

        all_metrics.append(metrics)
        print(
            f"  {model_name}: Acc={metrics['accuracy']:.4f} | F1w={metrics['f1_weighted']:.4f} | AUC={metrics['roc_auc']:.4f}"
        )

    if not all_metrics:
        print("  [ERROR] No models to evaluate!")
        return []

    best = max(all_metrics, key=lambda x: x["f1_weighted"])
    print(
        f"\n  ★ Best Model: {best['model_name']} (F1-weighted: {best['f1_weighted']:.4f})"
    )

    comparison = {
        "models": all_metrics,
        "best_model": best["model_name"],
        "best_metric": "f1_weighted",
        "best_value": best["f1_weighted"],
        "dataset": dataset_key,
    }

    comparison_path = os.path.join(OUTPUTS_DIR, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  [SAVED] Model comparison → {comparison_path}")

    csv_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    df_metrics = pd.DataFrame(
        [
            {
                "Model": m["model_name"],
                "Accuracy": m["accuracy"],
                "Precision(W)": m["precision_weighted"],
                "Recall(W)": m["recall_weighted"],
                "F1(W)": m["f1_weighted"],
                "F1(M)": m["f1_macro"],
                "AUC": m["roc_auc"],
                "Kappa": m["cohens_kappa"],
                "MCC": m["mcc"],
                "LogLoss": m["log_loss"],
                "CV Mean±Std": f"{m['cv_mean']:.4f}±{m['cv_std']:.4f}",
                "Time(s)": m["training_time"],
            }
            for m in all_metrics
        ]
    )
    df_metrics.to_csv(csv_path, index=False)
    print(f"  [SAVED] Comparison CSV → {csv_path}")

    print(f"\n{tabulate(df_metrics, headers='keys', tablefmt='grid', showindex=False)}")

    return all_metrics


def generate_evaluation_plots(
    all_metrics, results, X_test, y_test, label_mapping, dataset_key="primary"
):
    """Generate and save all evaluation visualization plots."""
    print(f"\n  Generating evaluation plots...")

    plt.style.use("dark_background")
    colors = [
        "#2563EB",
        "#EF4444",
        "#10B981",
        "#F59E0B",
        "#8B5CF6",
        "#F97316",
        "#06B6D4",
    ]

    model_names = [m["model_name"] for m in all_metrics]
    accuracies = [m["accuracy"] for m in all_metrics]
    f1_weighted = [m["f1_weighted"] for m in all_metrics]
    f1_macro = [m["f1_macro"] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.25
    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy", color=colors[0])
    bars2 = ax.bar(x, f1_weighted, width, label="F1 (Weighted)", color=colors[1])
    bars3 = ax.bar(x + width, f1_macro, width, label="F1 (Macro)", color=colors[2])
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy & F1 Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "model_comparison_bar.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  [SAVED] model_comparison_bar.png")

    for m in all_metrics:
        model_name = m["model_name"]
        cm = np.array(m["confusion_matrix"])
        class_labels = m.get("class_labels", [str(i) for i in range(cm.shape[0])])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axes[0],
            cbar_kws={"shrink": 0.8},
        )
        axes[0].set_title(f"{model_name} — Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        cm_norm = np.array(m["normalized_confusion_matrix"])
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axes[1],
            cbar_kws={"shrink": 0.8},
        )
        axes[1].set_title(f"{model_name} — Normalized Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  [SAVED] confusion_matrix_{model_name}.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, m in enumerate(all_metrics):
        roc_data = m.get("roc_data", {})
        for cls_name, cls_data in roc_data.items():
            fpr = cls_data["fpr"]
            tpr = cls_data["tpr"]
            auc_val = cls_data["auc"]
            ax.plot(
                fpr,
                tpr,
                label=f"{m['model_name']} - {cls_name} (AUC={auc_val:.3f})",
                alpha=0.7,
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "roc_curves_all.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  [SAVED] roc_curves_all.png")

    cv_data = []
    cv_labels = []
    for model_name, result in results.items():
        cv_scores = result.get("cv_scores", [])
        if cv_scores and len(cv_scores) > 1:
            cv_data.append(cv_scores)
            cv_labels.append(model_name)

    if cv_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        ax.set_ylabel("F1 Score (Weighted)")
        ax.set_title("Cross-Validation Score Distribution")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, "cv_boxplot.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"  [SAVED] cv_boxplot.png")

    tree_models = ["RandomForest", "XGBoost", "GradientBoosting"]
    for model_name in tree_models:
        result = results.get(model_name)
        if result and result.get("feature_importances"):
            fi_list = result["feature_importances"]
            top_n = min(20, len(fi_list))
            names = [fi["name"] for fi in fi_list[:top_n]]
            values = [fi["importance"] for fi in fi_list[:top_n]]

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(top_n), values[::-1], color=colors[0], alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(names[::-1], fontsize=8)
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"{model_name} — Top {top_n} Feature Importances")
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, f"feature_importance_{model_name}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            print(f"  [SAVED] feature_importance_{model_name}.png")

    training_times = [m["training_time"] for m in all_metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = np.argsort(training_times)[::-1]
    sorted_names = [model_names[i] for i in sorted_idx]
    sorted_times = [training_times[i] for i in sorted_idx]
    ax.barh(sorted_names, sorted_times, color=colors[3], alpha=0.8)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison")
    for i, v in enumerate(sorted_times):
        ax.text(v + 0.1, i, f"{v:.2f}s", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "training_time_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  [SAVED] training_time_comparison.png")

    print(f"  [DONE] All evaluation plots saved to {PLOTS_DIR}")
