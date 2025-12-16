#!/usr/bin/env python3
"""
DEAM Neural Network Cross-Validation Analysis
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DEAMNeuralNetworkCV:
    def __init__(self, csv_path: str | None = None):
        self.results = {}
        self.csv_path = csv_path

    def load_deam_data(self):
        print("=" * 80)
        print("DEAM Neural Network Cross-Validation")
        print("=" * 80)

        base_dir = Path(__file__).resolve().parent
        if self.csv_path is None:
            csv_path = base_dir / "data" / "deam_features_74.csv"
        else:
            csv_path = Path(self.csv_path)

        print(f"\nLoading features from: {csv_path}")

        if csv_path.exists():
            df = pd.read_csv(csv_path)

            possible_valence = ["valence", "valence_mean", "valence_avg"]
            possible_arousal = ["arousal", "arousal_mean", "arousal_avg"]

            valence_col = next((c for c in possible_valence if c in df.columns), None)
            arousal_col = next((c for c in possible_arousal if c in df.columns), None)

            if valence_col is None or arousal_col is None:
                raise ValueError(
                    f"Could not find valence/arousal columns. "
                    f"Looked for: {possible_valence} and {possible_arousal}"
                )

            drop_cols = {valence_col, arousal_col, "track_id", "song_id", "file_id", "filename"}
            feature_cols = [
                c for c in df.columns
                if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
            ]

            X = df[feature_cols].to_numpy(dtype=float)
            valence = df[valence_col].to_numpy(dtype=float)
            arousal = df[arousal_col].to_numpy(dtype=float)

            print(f"\nGot {X.shape[0]} tracks with {X.shape[1]} features")
            print(f"Valence range: {valence.min():.2f} to {valence.max():.2f}")
            print(f"Arousal range: {arousal.min():.2f} to {arousal.max():.2f}\n")

            return X, valence, arousal

        raise FileNotFoundError(f"Could not find DEAM feature CSV at: {csv_path}")

    def create_pipelines(self):
        def make_mlp():
            return MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            )

        pipeline_valence = Pipeline([("scaler", StandardScaler()), ("mlp", make_mlp())])
        pipeline_arousal = Pipeline([("scaler", StandardScaler()), ("mlp", make_mlp())])

        print("Neural Network: 128 -> 64 -> 32 layers, ReLU activation, early stopping\n")
        return pipeline_valence, pipeline_arousal

    def run_cross_validation(self, X, y, pipeline, target_name: str, n_folds: int = 5):
        print(f"\n{'=' * 80}")
        print(f"{n_folds}-Fold Cross-Validation: {target_name.upper()}")
        print(f"{'=' * 80}\n")

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        start_time = time.time()
        cv_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="r2", n_jobs=-1, verbose=1)
        cv_time = time.time() - start_time

        mean_r2, std_r2 = cv_scores.mean(), cv_scores.std()

        print(f"\nResults:")
        for i, score in enumerate(cv_scores, 1):
            print(f"  Fold {i}: {score:.4f}")
        print(f"\n  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"  Range: {cv_scores.min():.4f} to {cv_scores.max():.4f}")
        print(f"  Time: {cv_time:.1f}s\n")

        return {
            "scores": cv_scores,
            "mean": mean_r2,
            "std": std_r2,
            "min": cv_scores.min(),
            "max": cv_scores.max(),
            "time": cv_time,
        }

    def compare_to_single_split(self, X, y_valence, y_arousal):
        print(f"\n{'=' * 80}")
        print("Single 80/20 Train/Test Split (for comparison)")
        print(f"{'=' * 80}\n")

        X_train, X_test, y_val_train, y_val_test, y_ar_train, y_ar_test = train_test_split(
            X, y_valence, y_arousal, test_size=0.2, random_state=42
        )

        pipeline_val, pipeline_ar = self.create_pipelines()

        pipeline_val.fit(X_train, y_val_train)
        val_r2 = r2_score(y_val_test, pipeline_val.predict(X_test))

        pipeline_ar.fit(X_train, y_ar_train)
        ar_r2 = r2_score(y_ar_test, pipeline_ar.predict(X_test))

        print(f"  Valence R²: {val_r2:.4f}")
        print(f"  Arousal R²: {ar_r2:.4f}\n")

        return {"valence_r2": val_r2, "arousal_r2": ar_r2}

    def visualize_cv_results(self, valence_cv, arousal_cv, single_split):
        print(f"\n{'=' * 80}")
        print("Generating Visualization")
        print(f"{'=' * 80}\n")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "DEAM Neural Network: 5-Fold Cross-Validation Analysis",
            fontsize=14,
            fontweight="bold",
        )

        folds = np.arange(1, 6)

        # Plot 1: Valence CV scores
        ax1 = axes[0, 0]
        ax1.plot(
            folds,
            valence_cv["scores"],
            "o-",
            color="#3498db",
            linewidth=2,
            markersize=10,
            label="CV Scores",
        )
        ax1.axhline(
            y=valence_cv["mean"],
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {valence_cv['mean']:.4f}",
        )
        ax1.axhline(
            y=single_split["valence_r2"],
            color="#2ecc71",
            linestyle=":",
            linewidth=2,
            label=f"Single Split: {single_split['valence_r2']:.4f}",
        )
        ax1.fill_between(
            folds,
            valence_cv["mean"] - valence_cv["std"],
            valence_cv["mean"] + valence_cv["std"],
            alpha=0.2,
            color="#3498db",
        )

        ax1.set_xlabel("Fold", fontweight="bold")
        ax1.set_ylabel("R² Score", fontweight="bold")
        ax1.set_title("Valence Prediction: Cross-Validation", fontweight="bold")
        ax1.set_xticks(folds)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Arousal CV scores
        ax2 = axes[0, 1]
        ax2.plot(
            folds,
            arousal_cv["scores"],
            "o-",
            color="#9b59b6",
            linewidth=2,
            markersize=10,
            label="CV Scores",
        )
        ax2.axhline(
            y=arousal_cv["mean"],
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {arousal_cv['mean']:.4f}",
        )
        ax2.axhline(
            y=single_split["arousal_r2"],
            color="#2ecc71",
            linestyle=":",
            linewidth=2,
            label=f"Single Split: {single_split['arousal_r2']:.4f}",
        )
        ax2.fill_between(
            folds,
            arousal_cv["mean"] - arousal_cv["std"],
            arousal_cv["mean"] + arousal_cv["std"],
            alpha=0.2,
            color="#9b59b6",
        )

        ax2.set_xlabel("Fold", fontweight="bold")
        ax2.set_ylabel("R² Score", fontweight="bold")
        ax2.set_title("Arousal Prediction: Cross-Validation", fontweight="bold")
        ax2.set_xticks(folds)
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Plot 3: Box plot comparison
        ax3 = axes[1, 0]
        data = [valence_cv["scores"], arousal_cv["scores"]]
        bp = ax3.boxplot(
            data,
            labels=["Valence", "Arousal"],
            patch_artist=True,
            widths=0.6,
        )

        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#9b59b6")

        ax3.plot(
            1,
            single_split["valence_r2"],
            "r*",
            markersize=15,
            label="Single Split",
            zorder=3,
        )
        ax3.plot(2, single_split["arousal_r2"], "r*", markersize=15, zorder=3)

        ax3.set_ylabel("R² Score", fontweight="bold")
        ax3.set_title("Distribution of CV Scores", fontweight="bold")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)

        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis("off")

        summary_data = [
            ["Metric", "Valence", "Arousal"],
            ["", "", ""],
            ["CV Mean R²", f"{valence_cv['mean']:.4f}", f"{arousal_cv['mean']:.4f}"],
            ["CV Std Dev", f"{valence_cv['std']:.4f}", f"{arousal_cv['std']:.4f}"],
            ["CV Min R²", f"{valence_cv['min']:.4f}", f"{arousal_cv['min']:.4f}"],
            ["CV Max R²", f"{valence_cv['max']:.4f}", f"{arousal_cv['max']:.4f}"],
            ["", "", ""],
            [
                "Single Split R²",
                f"{single_split['valence_r2']:.4f}",
                f"{single_split['arousal_r2']:.4f}",
            ],
            ["", "", ""],
            [
                "Literature target (approx.)",
                "0.81",
                "0.83",
            ],
            ["", "", ""],
            [
                "Diff from literature",
                f"{(valence_cv['mean'] - 0.81):.4f}",
                f"{(arousal_cv['mean'] - 0.83):.4f}",
            ],
        ]

        table = ax4.table(
            cellText=summary_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.4, 0.3, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)

        for i in range(3):
            table[(0, i)].set_facecolor("#34495e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.tight_layout()
        plt.savefig("deam_neural_network_cv_results.png", dpi=300, bbox_inches="tight")
        print("\nVisualization saved as 'deam_neural_network_cv_results.png'")
        plt.show()


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "deam_features_74.csv"
    cv_analysis = DEAMNeuralNetworkCV(csv_path=str(data_path))

    X, y_valence, y_arousal = cv_analysis.load_deam_data()
    pipeline_valence, pipeline_arousal = cv_analysis.create_pipelines()

    valence_cv_results = cv_analysis.run_cross_validation(
        X, y_valence, pipeline_valence, "valence", n_folds=5
    )
    arousal_cv_results = cv_analysis.run_cross_validation(
        X, y_arousal, pipeline_arousal, "arousal", n_folds=5
    )
    single_split_results = cv_analysis.compare_to_single_split(X, y_valence, y_arousal)

    cv_analysis.visualize_cv_results(valence_cv_results, arousal_cv_results, single_split_results)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nValence: {valence_cv_results['mean']:.4f} ± {valence_cv_results['std']:.4f} (CV)")
    print(f"         {single_split_results['valence_r2']:.4f} (single split)")
    print(f"\nArousal: {arousal_cv_results['mean']:.4f} ± {arousal_cv_results['std']:.4f} (CV)")
    print(f"         {single_split_results['arousal_r2']:.4f} (single split)")
    print(f"\n{'=' * 80}")
    print("Analysis complete. See 'deam_neural_network_cv_results.png'")
    print("=" * 80)


if __name__ == "__main__":
    main()
