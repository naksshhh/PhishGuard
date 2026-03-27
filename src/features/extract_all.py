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
    html_features_path = PROCESSED_DIR / "mendeley_html_features.csv"
    
    if mendeley_meta_path.exists():
        mendeley_meta = pd.read_csv(mendeley_meta_path)
        logger.info(f"Loaded Mendeley metadata: {len(mendeley_meta)} rows")

        # ── Optimized HTML Extraction with Checkpointing ──
        # Check for existing progress
        existing_df = pd.DataFrame(columns=["html_filename"] + HTML_FEATURE_NAMES)
        if html_features_path.exists():
            existing_df = pd.read_csv(html_features_path)
            logger.info(f"Resuming: Found {len(existing_df)} already processed samples.")

        # Identify missing files
        unprocessed_meta = mendeley_meta[
            (mendeley_meta["html_exists"] == True) & 
            (~mendeley_meta["html_filename"].isin(existing_df["html_filename"]))
        ].copy()

        if len(unprocessed_meta) > 0:
            logger.info(f"Extracting HTML features for {len(unprocessed_meta)} remaining files...")
            
            # Process in large chunks to allow for Auto-Saving
            CHUNK_SIZE = 5000
            for i in range(0, len(unprocessed_meta), CHUNK_SIZE):
                chunk = unprocessed_meta.iloc[i : i + CHUNK_SIZE]
                logger.info(f"Processing chunk {i//CHUNK_SIZE + 1} ({len(chunk)} files)...")
                
                # Use the new parallel batch extractor
                chunk_features = extract_html_features_batch(chunk, HTML_DIR, n_jobs=-1)
                
                # Append to existing and save (Auto-Save)
                existing_df = pd.concat([existing_df, chunk_features], ignore_index=True)
                existing_df.to_csv(html_features_path, index=False)
                logger.info(f"Progress saved to {html_features_path} (Total: {len(existing_df)})")
        else:
            logger.info("All Mendeley HTML features are already extracted.")

        html_features = existing_df
    else:
        mendeley_meta = None

    # ── Split Processing logic ──
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
        logger.info(f"  Generating Final Feature Set: {split_name}")
        logger.info(f"{'=' * 50}")

        df = pd.read_csv(split_path)
        
        # 1. URL lexical features (Lexical features are fast enough)
        url_features = extract_features_batch(df["url"], show_progress=True)

        # 2. Merge URL features with labels
        result = pd.concat([
            df[["url", "label", "source"]].reset_index(drop=True),
            url_features.reset_index(drop=True),
        ], axis=1)

        # 3. Join with the pre-extracted HTML structural features
        if mendeley_meta is not None and html_features_path.exists():
            # url_to_html mapping from Mendeley
            url_to_html = mendeley_meta.set_index("url")["html_filename"].to_dict()
            result["html_filename"] = result["url"].map(url_to_html)

            # Merge
            result = result.merge(html_features, on="html_filename", how="left")
            result.drop(columns=["html_filename"], inplace=True, errors="ignore")

            # Fill missing HTML features (for URLs from non-Mendeley sources)
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
        logger.info(f"Saved {split_name} features to {output_path} ({len(result)} rows)")

    logger.info("\n✅ All features extracted and unified!")


if __name__ == "__main__":
    run_feature_extraction()
