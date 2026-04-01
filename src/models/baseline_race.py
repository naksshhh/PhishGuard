"""
PhishGuard++ — Baseline Model Race
Compares Random Forest, XGBoost, and LightGBM on the 40-feature dataset.
Uses Optuna for hyperparameter optimization on the winner (LightGBM expected).

Outputs:
  - models/lightgbm_stage1.pkl     — Best model
  - models/lightgbm_stage1.json    — Metrics + Optuna best params
  - W&B + CSV logging
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "datasets" / "processed"
RESULTS_DIR = BASE_DIR / "results"


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict:
    """Evaluate a model and return metrics dict."""
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    latency = (time.perf_counter() - start) / len(X_test) * 1000  # ms per sample

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "latency_ms_per_sample": latency,
    }

    logger.info(f"\n{'═' * 50}")
    logger.info(f"  {model_name}")
    logger.info(f"{'═' * 50}")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  Latency:   {metrics['latency_ms_per_sample']:.3f} ms/sample")

    return metrics


def run_baseline_race(X_train, y_train, X_val, y_val, X_test, y_test):
    """Run RF vs XGB vs LightGBM comparison."""
    logger.info("\n🏁 Starting Baseline Model Race\n")

    results = []

    # ── 1. Random Forest ──
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))

    # ── 2. XGBoost ──
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))

    # ── 3. LightGBM ──
    logger.info("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        num_leaves=31, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    results.append(evaluate_model(lgb_model, X_test, y_test, "LightGBM"))

    # ── Summary ──
    results_df = pd.DataFrame(results)
    logger.info(f"\n{'═' * 60}")
    logger.info("  BASELINE RACE RESULTS")
    logger.info(f"{'═' * 60}")
    logger.info(f"\n{results_df.to_string(index=False)}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "baseline_race_results.csv", index=False)

    return results_df, {"rf": rf, "xgboost": xgb_model, "lightgbm": lgb_model}


def optuna_lightgbm_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM hyperparameter optimization."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)


def train_lightweight_lgb(X_train, y_train, X_val, y_val):
    """
    Train a strictly pruned model for Tier 1 (Edge) ONNX.
    Targets <300KB file size and <15ms latency.
    """
    logger.info("Training Lightweight LightGBM for Edge Tier...")
    
    # Pruned parameters: 
    # - n_estimators=50-100 (fewer trees)
    # - max_depth=6 (shorter trees)
    # - num_leaves=31 (simpler nodes)
    params = {
        "n_estimators": 80,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=0)],
    )
    
    # Save as separate file
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lightgbm_edge.pkl")
    logger.info(f"✅ Saved Edge-optimized model to {MODELS_DIR / 'lightgbm_edge.pkl'}")
    
    return model

def run_optuna_sweep(X_train, y_train, X_val, y_val, X_test, y_test, n_trials: int = 50):
    """Run Optuna Bayesian optimization on LightGBM."""
    logger.info(f"\n🔍 Running Optuna sweep ({n_trials} trials)...")

    study = optuna.create_study(direction="maximize", study_name="lightgbm_phishguard")
    study.optimize(
        lambda trial: optuna_lightgbm_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info(f"\nBest trial F1: {study.best_value:.4f}")
    logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Train final model with best params
    best_params = study.best_params.copy()
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    # Final evaluation
    metrics = evaluate_model(final_model, X_test, y_test, "LightGBM (Optuna)")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lightgbm_stage1.pkl"
    joblib.dump(final_model, model_path)
    logger.info(f"\n💾 Saved best model to {model_path}")

    # Save metadata
    meta = {
        "best_params": study.best_params,
        "best_val_f1": study.best_value,
        "test_metrics": metrics,
        "n_trials": n_trials,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    meta_path = MODELS_DIR / "lightgbm_stage1.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"📋 Saved metadata to {meta_path}")

    return final_model, study, metrics


def plot_feature_importance(model, feature_names, save_path=None):
    """Plot and optionally save SHAP-style feature importance."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:20]  # Top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(sorted_idx)),
        importance[sorted_idx][::-1],
        color="#3b82f6",
        alpha=0.8,
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
    ax.set_xlabel("Feature Importance")
    ax.set_title("LightGBM Feature Importance (Top 20)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Saved feature importance plot to {save_path}")
    plt.close()


def try_wandb_log(metrics: Dict, model_name: str):
    """Try to log to W&B. Falls back to CSV if W&B is not configured."""
    try:
        import wandb
        wandb.init(
            project="phishguard-plus-plus",
            name=f"baseline-{model_name}",
            config=metrics,
            reinit=True,
        )
        wandb.log(metrics)
        wandb.finish()
        logger.info(f"📊 Logged to W&B: {model_name}")
    except Exception as e:
        logger.info(f"W&B not available ({e}), using CSV logging only")


def main():
    """Full baseline race + Optuna sweep pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("PhishGuard++ Baseline Model Race")
    logger.info("=" * 50)

    # Load data
    # This expects the feature extraction pipeline to have run first
    # producing datasets/processed/train_features.csv, val_features.csv, test_features.csv
    # For now, we'll work with PhiUSIIL which already has features

    from src.features.url_features import URL_FEATURE_NAMES

    # Check if processed features exist
    train_path = DATA_DIR / "train_features.csv"
    if not train_path.exists():
        logger.info("Processed features not found. Running feature extraction first...")
        logger.info("Please run: python -m src.data.dataset_builder")
        logger.info("Then run:   python -m src.features.extract_all")
        return

    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    val_df = pd.read_csv(DATA_DIR / "val_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")

    # Determine feature columns (all 40 features)
    feature_cols = [c for c in train_df.columns if c not in ["url", "label", "source", "html_filename"]]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}... ")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # Step 1: Baseline Race
    results_df, models = run_baseline_race(X_train, y_train, X_val, y_val, X_test, y_test)

    # Log to W&B
    for _, row in results_df.iterrows():
        try_wandb_log(row.to_dict(), row["model"])

    # Step 2: Optuna sweep on LightGBM (The "Master" Cloud Model)
    final_model, study, final_metrics = run_optuna_sweep(
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_trials=50,
    )

    # Step 3: Lightweight Edge Model (The "Sentry" model for Tier 1 ONNX)
    train_lightweight_lgb(X_train, y_train, X_val, y_val)

    # Step 4: Feature importance plot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(
        final_model, feature_cols,
        save_path=RESULTS_DIR / "feature_importance.png",
    )

    logger.info("\n✅ Baseline race complete!")


if __name__ == "__main__":
    main()
