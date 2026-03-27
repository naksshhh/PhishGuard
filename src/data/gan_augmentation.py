"""
PhishGuard++ — GAN Data Augmentation
Uses CTGAN (from SDV) to generate synthetic phishing samples
to balance the dataset and improve model robustness.

Target: Generate 20,000 synthetic phishing samples.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "models"


def run_gan_augmentation(n_synthetic: int = 20000):
    """Train CTGAN on existing phishing samples and generate new ones."""
    logger.info("Starting GAN data augmentation (CTGAN)...")

    # Load train features (to learn patterns)
    train_path = DATA_DIR / "train_features.csv"
    if not train_path.exists():
        logger.error(f"Train features not found at {train_path}. Run feature extraction first.")
        return

    df = pd.read_csv(train_path)
    
    # Filter for phishing samples only (we want to augment the minority/critical class)
    phish_df = df[df["label"] == 1].copy()
    
    # Drop non-feature columns for GAN training
    cols_to_drop = ["url", "label", "source"]
    train_data = phish_df.drop(columns=cols_to_drop, errors="ignore")
    
    logger.info(f"Training CTGAN on {len(train_data)} real phishing samples...")

    # Create metadata for SDV
    metadata = Metadata.detect_from_dataframe(
        data=train_data,
        table_name='phishing_features'
    )

    # Initialize and train CTGAN
    # Note: For demo/speed, we use fewer epochs. In production, 100+ is better.
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_rounding=False,
        epochs=50,
        verbose=True,
        cuda=True  # Ensure GPU is used
    )
    
    synthesizer.fit(train_data)

    # Generate synthetic samples
    logger.info(f"Generating {n_synthetic} synthetic phishing samples...")
    synthetic_df = synthesizer.sample(num_rows=n_synthetic)

    # Add labels and source info
    synthetic_df["label"] = 1
    synthetic_df["source"] = "gan_synthetic"
    synthetic_df["url"] = "synthetic_url" # Placeholder

    # Save synthetic data
    output_path = DATA_DIR / "synthetic_phish_features.csv"
    synthetic_df.to_csv(output_path, index=False)
    logger.info(f"Saved synthetic data to {output_path}")

    # Save the synthesizer model
    synthesizer_path = MODELS_DIR / "ctgan_synthesizer.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    synthesizer.save(synthesizer_path)
    logger.info(f"Saved CTGAN model to {synthesizer_path}")

    return synthetic_df


if __name__ == "__main__":
    run_gan_augmentation()
