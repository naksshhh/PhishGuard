"""
PhishGuard++ — Benchmark & Ablation Study
Runs latency profiling, computes per-tier metrics, and performs
an 8-config ablation on the held-out test set.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "datasets" / "processed"


def load_test_data():
    """Load the held-out test set with features."""
    path = DATA_DIR / "test_features.csv"
    if not path.exists():
        logger.error(f"Test features not found at {path}.")
        return None, None, None
    df = pd.read_csv(path)
    from src.features.url_features import URL_FEATURE_NAMES
    from src.features.html_features import HTML_FEATURE_NAMES
    feature_cols = URL_FEATURE_NAMES + HTML_FEATURE_NAMES
    X = df[feature_cols].fillna(0)
    y = df["label"]
    return X, y, feature_cols


def benchmark_latency(model, X, n_runs: int = 100):
    """Profile inference latency per sample."""
    sample = X.iloc[:1]
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - start) * 1000)  # ms
    return {
        "mean_ms": np.mean(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }


def compute_metrics(y_true, y_pred):
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": (((y_pred == 1) & (y_true == 0)).sum() / (y_true == 0).sum()
                if (y_true == 0).sum() > 0 else 0),
    }


def run_ablation():
    """Run 8-config ablation study on the test set."""
    X, y, feature_cols = load_test_data()
    if X is None:
        return

    from src.features.url_features import URL_FEATURE_NAMES
    from src.features.html_features import HTML_FEATURE_NAMES

    lgb_path = MODELS_DIR / "lightgbm_stage1.pkl"
    if not lgb_path.exists():
        logger.error("LightGBM model not found. Run baseline_race.py first.")
        return

    model = joblib.load(lgb_path)

    configs = {
        "full_40_features": feature_cols,
        "url_only_20": URL_FEATURE_NAMES,
        "html_only_20": HTML_FEATURE_NAMES,
        "top_10_url": URL_FEATURE_NAMES[:10],
        "top_10_html": HTML_FEATURE_NAMES[:10],
        "lexical_5": URL_FEATURE_NAMES[:5],
        "structural_5": HTML_FEATURE_NAMES[:5],
        "minimal_3": ["url_length", "entropy_domain", "form_action_external"],
    }

    results = []
    for name, cols in configs.items():
        # Pad missing features with 0 to match model's expected input
        X_config = pd.DataFrame(0, index=X.index, columns=feature_cols)
        for col in cols:
            if col in X.columns:
                X_config[col] = X[col]

        predictions = model.predict(X_config)
        metrics = compute_metrics(y, predictions)
        metrics["config"] = name
        metrics["n_features"] = len(cols)

        # Latency
        latency = benchmark_latency(model, X_config)
        metrics.update(latency)

        results.append(metrics)
        logger.info(f"{name}: F1={metrics['f1']:.4f}, Latency(p95)={metrics['p95_ms']:.2f}ms")

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / "ablation_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved ablation results to {out_path}")
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    run_ablation()
