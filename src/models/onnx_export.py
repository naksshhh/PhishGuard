"""
PhishGuard++ — ONNX Export & Quantization
Converts trained LightGBM models into ONNX format for 
on-device inference in the Chrome extension.
"""

import logging
from pathlib import Path

import joblib
import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime.quantization import QuantType, quantize_dynamic
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


def export_lgb_to_onnx(model_path: Path, output_name: str = "phishguard_edge.onnx"):
    """Convert .pkl LightGBM model to ONNX with INT8 dynamic quantization."""
    logger.info(f"Loading LightGBM model from {model_path}...")
    
    # Load model
    model = joblib.load(model_path)
    
    # Get number of features
    # (Assuming we used 40 features as per plan)
    n_features = 40 # Standard check, can be derived from model.n_features_
    
    logger.info(f"Converting to ONNX (n_features={n_features})...")
    
    # Convert to ONNX
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=12)
    
    # Save raw ONNX
    raw_path = MODELS_DIR / f"raw_{output_name}"
    onnx.save_model(onnx_model, raw_path)
    logger.info(f"Saved raw ONNX to {raw_path}")
    
    # Apply Dynamic Quantization (INT8) for browser performance
    logger.info("Applying dynamic quantization...")
    quantized_path = MODELS_DIR / output_name
    
    quantize_dynamic(
        model_input=raw_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8
    )
    
    # Clean up raw file
    if raw_path.exists():
        raw_path.unlink()
        
    logger.info(f"✅ Exported and quantized model to {quantized_path}")
    logger.info(f"   Size: {quantized_path.stat().st_size / 1024:.2f} KB (Target: <300KB)")
    
    return quantized_path


if __name__ == "__main__":
    # This expects models/lightgbm_edge.pkl to exist (from baseline_race.py)
    lgb_pkl = MODELS_DIR / "lightgbm_edge.pkl"
    if lgb_pkl.exists():
        export_lgb_to_onnx(lgb_pkl)
    else:
        # Fallback to stage1 if edge-specific model isn't built
        lgb_pkl = MODELS_DIR / "lightgbm_stage1.pkl"
        if lgb_pkl.exists():
            export_lgb_to_onnx(lgb_pkl)
        else:
            logger.error(f"Model not found at {lgb_pkl}. Run baseline_race.py first.")
