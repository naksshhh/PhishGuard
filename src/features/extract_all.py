"""
PhishGuard++ — Feature Extraction Orchestrator
Runs URL + HTML feature extraction on the unified dataset
and produces train/val/test feature CSVs ready for model training.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
HTML_DIR = DATA_DIR / "Mendeley phishing dataset"


def run_feature_extraction():
    """Extract all 40 features (20 URL + 20 HTML) for every split."""
    from src.features.url_features import extract_features_batch, URL_FEATURE_NAMES
    from src.features.html_features import (
        extract_html_features_batch, HTML_FEATURE_NAMES,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load Mendeley metadata for HTML features
    mendeley_meta_path = PROCESSED_DIR / "mendeley_metadata.csv"
    mendeley_meta = None
    if mendeley_meta_path.exists():
        mendeley_meta = pd.read_csv(mendeley_meta_path)
        logger.info(f"Loaded Mendeley metadata: {len(mendeley_meta)} rows")

        # Extract HTML features for Mendeley entries
        html_features_path = PROCESSED_DIR / "mendeley_html_features.csv"
        if not html_features_path.exists():
            logger.info("Extracting HTML features from Mendeley dataset...")
            html_features = extract_html_features_batch(mendeley_meta, HTML_DIR)
            html_features.to_csv(html_features_path, index=False)
            logger.info(f"Saved HTML features to {html_features_path}")
        else:
            html_features = pd.read_csv(html_features_path)
            logger.info(f"Loaded cached HTML features: {len(html_features)} rows")

    # Process each split
    for split_name in ["train", "val", "test"]:
        split_path = PROCESSED_DIR / f"{split_name}_urls.csv"
        output_path = PROCESSED_DIR / f"{split_name}_features.csv"

        if not split_path.exists():
            logger.warning(f"Split file not found: {split_path}. Run dataset_builder.py first.")
            continue

        if output_path.exists():
            logger.info(f"Features already exist for {split_name}, skipping. Delete to regenerate.")
            continue

        logger.info(f"\n{'=' * 50}")
        logger.info(f"  Processing {split_name} split")
        logger.info(f"{'=' * 50}")

        df = pd.read_csv(split_path)
        logger.info(f"Loaded {len(df)} URLs")

        # Extract URL features
        url_features = extract_features_batch(df["url"], show_progress=True)

        # Merge URL features with labels
        result = pd.concat([
            df[["url", "label", "source"]].reset_index(drop=True),
            url_features.reset_index(drop=True),
        ], axis=1)

        # Try to join HTML features for Mendeley entries
        if mendeley_meta is not None and html_features_path.exists():
            html_feats = pd.read_csv(html_features_path)

            # Create URL → HTML features mapping from Mendeley
            if "html_filename" in mendeley_meta.columns:
                url_to_html = mendeley_meta.set_index("url")["html_filename"].to_dict()
                result["html_filename"] = result["url"].map(url_to_html)

                # Merge HTML features
                result = result.merge(
                    html_feats, on="html_filename", how="left",
                )
                result.drop(columns=["html_filename"], inplace=True, errors="ignore")

                # Fill NaN HTML features with 0 (URLs without HTML)
                for col in HTML_FEATURE_NAMES:
                    if col in result.columns:
                        result[col] = result[col].fillna(0)
                    else:
                        result[col] = 0

        # Ensure all 40 features exist
        for col in URL_FEATURE_NAMES + HTML_FEATURE_NAMES:
            if col not in result.columns:
                result[col] = 0

        result.to_csv(output_path, index=False)
        logger.info(f"Saved {split_name} features to {output_path} ({len(result)} rows, {result.shape[1]} cols)")

    logger.info("\n✅ Feature extraction complete!")


if __name__ == "__main__":
    run_feature_extraction()
