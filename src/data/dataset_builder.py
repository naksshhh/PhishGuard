"""
PhishGuard++ — Dataset Builder
Merges all 5 data sources into a unified corpus for model training.

Datasets:
  1. Kaggle UCI           — Pre-computed features + label (Result column)
  2. PhiUSIIL             — URLs + 50+ features + label
  3. PhishTank            — Verified phishing URLs (all label=1)
  4. Tranco Top 1M        — Legitimate domains (all label=0)
  5. Mendeley Phishing    — URLs + HTML files + labels from SQL

Output:
  datasets/unified_urls.csv     — url, label, source
  datasets/unified_features.csv — url, 40 engineered features, label
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = BASE_DIR / "datasets"
OUTPUT_DIR = DATA_DIR / "processed"


def load_kaggle_uci() -> pd.DataFrame:
    """Load Kaggle UCI phishing dataset.
    Columns are pre-computed binary features + 'Result' label (-1=phishing, 1=legit).
    We keep only the label; features are re-extracted from URLs in PhiUSIIL.
    """
    path = DATA_DIR / "Kaggle_UCI.csv"
    logger.info(f"Loading Kaggle UCI from {path}")
    df = pd.read_csv(path)

    # The 'Result' column: -1 = phishing, 1 = legitimate
    # We don't have raw URLs here, so we skip this for URL merging
    # but keep it for the ablation baseline (pre-computed features)
    df["label"] = df["Result"].map({-1: 1, 1: 0})  # 1=phishing, 0=legit
    df["source"] = "kaggle_uci"
    logger.info(f"  → {len(df)} rows ({df['label'].sum()} phishing, {(df['label']==0).sum()} legit)")
    return df


def load_phiusiil() -> pd.DataFrame:
    """Load PhiUSIIL dataset — has URLs + 50+ pre-computed features + label."""
    path = DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    logger.info(f"Loading PhiUSIIL from {path}")
    df = pd.read_csv(path)

    # 'label' column: 1 = phishing, 0 = legitimate
    df["source"] = "phiusiil"
    logger.info(f"  → {len(df)} rows ({df['label'].sum()} phishing, {(df['label']==0).sum()} legit)")
    return df


def load_phishtank() -> pd.DataFrame:
    """Load PhishTank — all verified phishing URLs."""
    path = DATA_DIR / "PhishTank.csv"
    logger.info(f"Loading PhishTank from {path}")
    df = pd.read_csv(path)

    # Keep only essential columns
    df = df[["url"]].copy()
    df["label"] = 1  # All phishing
    df["source"] = "phishtank"
    df = df.drop_duplicates(subset=["url"])
    logger.info(f"  → {len(df)} phishing URLs")
    return df


def load_tranco() -> pd.DataFrame:
    """Load Tranco Top 1M — popular legitimate domains.
    Format: rank,domain (no header in some versions).
    """
    path = DATA_DIR / "Tranco_top_1m.csv"
    logger.info(f"Loading Tranco from {path}")

    # Try to detect if there's a header
    df = pd.read_csv(path, header=None, names=["rank", "domain"])

    # Convert domain to full URL for feature extraction
    df["url"] = "https://" + df["domain"]
    df["label"] = 0  # All legitimate
    df["source"] = "tranco"
    df = df[["url", "label", "source"]].copy()
    logger.info(f"  → {len(df)} legitimate domains")
    return df


def load_mendeley_sql() -> pd.DataFrame:
    """Parse the Mendeley SQL dump to extract URL → label → HTML filename mapping."""
    sql_path = DATA_DIR / "Mendeley phishing dataset" / "index.sql"
    logger.info(f"Loading Mendeley SQL from {sql_path}")

    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Parse INSERT statements
    # Pattern: (rec_id, 'url', 'filename', label, 'timestamp')
    pattern = r"\((\d+),\s*'([^']*)',\s*'([^']*)',\s*(\d+),\s*'([^']*)'\)"
    matches = re.findall(pattern, content)

    records = []
    for match in matches:
        rec_id, url, filename, label, timestamp = match
        records.append({
            "rec_id": int(rec_id),
            "url": url,
            "html_filename": filename,
            "label": int(label),  # 0=legit, 1=phishing
            "timestamp": timestamp,
        })

    df = pd.DataFrame(records)
    df["source"] = "mendeley"

    # Verify HTML files exist
    html_dir = DATA_DIR / "Mendeley phishing dataset"
    df["html_exists"] = df["html_filename"].apply(
        lambda f: (html_dir / f).exists()
    )

    existing = df["html_exists"].sum()
    logger.info(f"  → {len(df)} rows ({df['label'].sum()} phishing, {(df['label']==0).sum()} legit)")
    logger.info(f"  → {existing}/{len(df)} HTML files found on disk")
    return df


def build_unified_url_corpus(
    max_tranco: int = 50_000,
    max_phishtank: int = 30_000,
) -> pd.DataFrame:
    """
    Build unified URL corpus from all sources.

    Returns DataFrame with columns: url, label, source
    """
    logger.info("=" * 60)
    logger.info("Building unified URL corpus")
    logger.info("=" * 60)

    frames = []

    # 1. PhiUSIIL — has both URLs and labels
    phiusiil = load_phiusiil()
    frames.append(phiusiil[["URL", "label", "source"]].rename(columns={"URL": "url"}))

    # 2. PhishTank — all phishing
    phishtank = load_phishtank()
    if len(phishtank) > max_phishtank:
        phishtank = phishtank.sample(n=max_phishtank, random_state=42)
    frames.append(phishtank[["url", "label", "source"]])

    # 3. Tranco — all legitimate
    tranco = load_tranco()
    if len(tranco) > max_tranco:
        tranco = tranco.head(max_tranco)  # Top-ranked domains are most legitimate
    frames.append(tranco[["url", "label", "source"]])

    # 4. Mendeley — has URLs and labels
    mendeley = load_mendeley_sql()
    frames.append(mendeley[["url", "label", "source"]])

    # Merge
    unified = pd.concat(frames, ignore_index=True)

    # Deduplicate by URL (keep first occurrence)
    before = len(unified)
    unified = unified.drop_duplicates(subset=["url"], keep="first")
    logger.info(f"Deduplication: {before} → {len(unified)} ({before - len(unified)} duplicates removed)")

    # Stats
    logger.info("\n── Unified Corpus Stats ──")
    logger.info(f"Total samples:  {len(unified)}")
    logger.info(f"Phishing (1):   {(unified['label']==1).sum()}")
    logger.info(f"Legitimate (0): {(unified['label']==0).sum()}")
    logger.info(f"Balance ratio:  {(unified['label']==1).sum() / len(unified):.2%}")
    logger.info(f"\nSources:\n{unified['source'].value_counts().to_string()}")

    return unified


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Stratified train/val/test split."""
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df["label"], random_state=random_state,
    )

    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=random_state,
    )

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/len(df):.1%})")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/len(df):.1%})")

    return {"train": train_df, "val": val_df, "test": test_df}


def main():
    """Build and save the unified dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build unified URL corpus
    unified = build_unified_url_corpus()

    # Save full corpus
    out_path = OUTPUT_DIR / "unified_urls.csv"
    unified.to_csv(out_path, index=False)
    logger.info(f"\nSaved unified corpus to {out_path}")

    # Create splits
    splits = create_train_val_test_split(unified)
    for split_name, split_df in splits.items():
        split_path = OUTPUT_DIR / f"{split_name}_urls.csv"
        split_df.to_csv(split_path, index=False)
        logger.info(f"Saved {split_name} to {split_path}")

    # Save Mendeley metadata separately (for HTML feature extraction)
    mendeley = load_mendeley_sql()
    mendeley_path = OUTPUT_DIR / "mendeley_metadata.csv"
    mendeley.to_csv(mendeley_path, index=False)
    logger.info(f"Saved Mendeley metadata to {mendeley_path}")

    logger.info("\n✅ Dataset building complete!")


if __name__ == "__main__":
    main()
