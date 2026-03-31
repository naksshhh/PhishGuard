import shap
import joblib
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict

# Import feature lists to stay consistent
try:
    from src.features.url_features import URL_FEATURE_NAMES, HUMAN_READABLE_NAMES as URL_HUMAN
    from src.features.html_features import HTML_FEATURE_NAMES, HUMAN_READABLE_NAMES as HTML_HUMAN
except ImportError:
    # Fallback if imports fail during standalone execution
    URL_FEATURE_NAMES = []
    HTML_FEATURE_NAMES = []
    URL_HUMAN = {}
    HTML_HUMAN = {}

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lightgbm_stage1.pkl"

# Combined feature list and human-readable map
FEATURE_ORDER = URL_FEATURE_NAMES + HTML_FEATURE_NAMES
HUMAN_MAP = {**URL_HUMAN, **HTML_HUMAN}

class PhishExplainer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PhishExplainer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        try:
            if not MODEL_PATH.exists():
                logger.error(f"Model not found at {MODEL_PATH}")
                self.model = None
                self.explainer = None
                return
                
            self.model = joblib.load(MODEL_PATH)
            # TreeExplainer is perfect for LightGBM
            self.explainer = shap.TreeExplainer(self.model)
            self._initialized = True
            logger.info("SHAP Explainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP Explainer: {e}")
            self.model = None
            self.explainer = None

    def explain(self, features: Dict[str, float], top_n: int = 3) -> str:
        """
        Produce a human-readable explanation for a single prediction.
        
        Args:
            features: Dictionary of 40 features
            top_n: Number of top reasons to return
            
        Returns:
            A string containing bulleted reasons
        """
        if not self.explainer:
            return "Explainability engine offline."
            
        try:
            # Ensure features are in the correct order for the model
            df = pd.DataFrame([features], columns=FEATURE_ORDER)
            
            # Calculate SHAP values
            # For LightGBM binary classifier, explainer returns (N_SAMPLES, N_FEATURES)
            # or a list [shap_0, shap_1] for multiclass. 
            # In latest versions, it usually returns the contribution to the log-odds (class 1).
            shap_values = self.explainer.shap_values(df)
            
            # Handle list output (binary/multiclass consistency)
            if isinstance(shap_values, list):
                # Typically index 1 is the 'phish' class contribution
                vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                vals = shap_values[0]

            # Map values to names and sort
            feature_importance = []
            for i, val in enumerate(vals):
                if val > 0: # Only care about features pushing towards 'PHISH'
                    feat_name = FEATURE_ORDER[i]
                    feature_importance.append({
                        "name": feat_name,
                        "value": val,
                        "readable": HUMAN_MAP.get(feat_name, feat_name)
                    })
            
            # Sort by highest contribution
            feature_importance.sort(key=lambda x: x["value"], reverse=True)
            
            if not feature_importance:
                return "No specific high-risk structural patterns detected."
                
            # Formatting as bulleted list
            reasons = [f"• {item['readable']}" for item in feature_importance[:top_n]]
            return "\n".join(reasons)
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return "Pattern analysis failed."

# Global Explainer
explainer = PhishExplainer()

def get_phish_explanation(features: Dict[str, float]) -> str:
    """Helper function to explain a phish prediction."""
    return explainer.explain(features)
