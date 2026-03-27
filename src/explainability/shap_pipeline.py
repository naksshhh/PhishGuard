"""
PhishGuard++ — SHAP Explainability Pipeline
Generates human-readable explanations of phishing verdicts
using TreeSHAP for tabular models (XGBoost/LightGBM).
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "datasets" / "processed"

# Human-readable feature name mapping (synced with url_features + html_features)
FEATURE_EXPLANATIONS = {
    "url_length": "URL is unusually long",
    "domain_length": "Domain name is unusually long",
    "n_subdomains": "URL has many subdomains",
    "entropy_domain": "Domain has high randomness (entropy)",
    "has_ip_as_domain": "URL uses an IP address instead of a domain",
    "suspicious_keyword_count": "URL contains suspicious keywords (login, verify, etc.)",
    "punycode_detected": "URL uses internationalized encoding (punycode)",
    "https_present": "URL uses HTTPS",
    "form_action_external": "Page sends form data to an external site",
    "iframe_count": "Page embeds hidden frames",
    "hidden_input_count": "Page has hidden fields",
    "login_form_present": "Page has a login/password form",
    "external_link_ratio": "Most links point to other websites",
    "script_count": "Page loads many scripts",
    "meta_refresh_present": "Page redirects automatically",
    "title_domain_mismatch": "Page title doesn't match the domain",
    "favicon_external": "Favicon loads from another domain",
}


def explain_prediction(
    model_path: Path,
    feature_values: np.ndarray,
    feature_names: list,
    top_k: int = 5,
) -> list[dict]:
    """
    Generate SHAP explanations for a single prediction.

    Returns:
        List of top-k features with their SHAP values and human-readable text.
    """
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_values.reshape(1, -1))

    # For binary classifiers, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Phishing class

    sv = shap_values[0]
    indices = np.argsort(np.abs(sv))[::-1][:top_k]

    explanations = []
    for idx in indices:
        name = feature_names[idx]
        explanations.append({
            "feature": name,
            "shap_value": float(sv[idx]),
            "direction": "increases risk" if sv[idx] > 0 else "decreases risk",
            "human_text": FEATURE_EXPLANATIONS.get(name, name.replace("_", " ").title()),
        })

    return explanations


def explain_batch(model_path: Path, X: pd.DataFrame, top_k: int = 5):
    """Generate SHAP explanations for a batch of predictions."""
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(f"Generated SHAP values for {len(X)} samples")
    return shap_values


if __name__ == "__main__":
    # Quick demo (requires a trained model)
    lgb_pkl = MODELS_DIR / "lightgbm_stage1.pkl"
    if lgb_pkl.exists():
        test_path = DATA_DIR / "test_features.csv"
        if test_path.exists():
            df = pd.read_csv(test_path)
            from src.features.url_features import URL_FEATURE_NAMES
            from src.features.html_features import HTML_FEATURE_NAMES
            feature_cols = URL_FEATURE_NAMES + HTML_FEATURE_NAMES
            sample = df[feature_cols].iloc[0].values
            result = explain_prediction(lgb_pkl, sample, feature_cols)
            for r in result:
                print(f"  {r['human_text']}: {r['direction']} (SHAP={r['shap_value']:.4f})")
    else:
        logger.info("No trained model. Run baseline_race.py first.")
