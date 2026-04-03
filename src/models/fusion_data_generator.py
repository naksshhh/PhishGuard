import logging
import asyncio
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "models"

from src.models.inference_pipeline import orchestrator

# ── Branch Score Collectors ──────────────────────────────────

async def get_tabular_score(features: dict):
    """Tier 2 Cloud (LightGBM)"""
    return await orchestrator.get_tabular_score(features)

async def get_semantic_url_score(url: str):
    """Tier 3 PhishBERT"""
    score, _ = await orchestrator.get_phishbert_score(url)
    return score

async def get_semantic_html_score(html_excerpt: str):
    """Tier 3 CodeBERT"""
    return await orchestrator.get_codebert_score(html_excerpt)

async def get_visual_score(image_path: Path):
    """Tier 3 EfficientNet"""
    return await orchestrator.get_efficientnet_score_from_path(image_path)

# ─────────────────────────────────────────────────────────────

async def generate_fusion_data(limit: int = 10000):
    """
    Increments the fusion meta-dataset by processing new samples from the full data pool.
    """
    output_path = DATA_DIR / "fusion_train_v2.csv"
    existing_urls = set()
    
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        if "url" in existing_df.columns:
            existing_urls = set(existing_df["url"].tolist())
        else:
            # If earlier version didn't save URL, we might need it for deduplication
            # For now, if no URL column, we'll assume we can't reliably append and start fresh or skip
            logger.warning("Existing CSV has no 'url' column. Cannot safely append. Starting fresh.")
            existing_urls = set()

    # 1. Load full pool
    train_path = DATA_DIR / "train_features.csv"
    val_path = DATA_DIR / "val_features.csv"
    meta_path = DATA_DIR / "mendeley_metadata.csv"

    if not all(p.exists() for p in [train_path, val_path, meta_path]):
        logger.error("Required feature/metadata files missing.")
        return

    logger.info("Loading dataset pools...")
    pool_df = pd.concat([pd.read_csv(train_path), pd.read_csv(val_path)])
    meta_df = pd.read_csv(meta_path)[["url", "html_filename"]]
    meta_df = meta_df.drop_duplicates(subset=["url"])
    
    df = pd.merge(pool_df, meta_df, on="url", how="inner")
    df = df.dropna(subset=["html_filename"])

    # 2. Filter out already processed
    new_df = df[~df["url"].isin(existing_urls)]
    
    if len(new_df) == 0:
        logger.info("No new samples to process.")
        return

    # 3. Sample new batch
    target_new = min(limit, len(new_df))
    batch_df = new_df.sample(n=target_new, random_state=42)
    
    logger.info(f"Targeting {target_new} new samples (Existing: {len(existing_urls)}). Total will be {len(existing_urls) + target_new}.")
    
    results = []
    
    # Open file in append mode
    file_exists = output_path.exists()
    
    for idx, row in tqdm(batch_df.iterrows(), total=target_new):
        try:
            # 1. Tabular Branch
            tabular_score = await get_tabular_score(row.to_dict())
            
            # 2. PhishBERT URL Branch
            url_score = await get_semantic_url_score(row["url"])
            
            # 3. CodeBERT HTML Branch
            html_file = BASE_DIR / "datasets" / "Mendeley phishing dataset" / row["html_filename"]
            html_text = ""
            if html_file.exists():
                with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                    html_text = f.read(2048)
            
            html_score, html_mask = await get_semantic_html_score(html_text)
            
            # 4. EfficientNet Visual Branch
            img_path = BASE_DIR / "datasets" / "screenshots" / row["html_filename"].replace(".html", ".jpg")
            visual_score, visual_mask = await get_visual_score(img_path)
            
            res = {
                "url": row["url"],
                "tabular_score": tabular_score,
                "url_score": url_score,
                "html_score": html_score,
                "html_mask": html_mask,
                "visual_score": visual_score,
                "visual_mask": visual_mask,
                "label": row["label"]
            }
            results.append(res)

            # Periodic checkpoint (every 50 samples)
            if len(results) >= 50:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
                results = []

        except Exception as e:
            logger.error(f"Error processing {row['url']}: {e}")

    # Final write for remaining
    if results:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

    logger.info(f"✅ Incremental update complete. Meta-dataset now at {output_path}")

if __name__ == "__main__":
    import sys
    limit = 10000
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    asyncio.run(generate_fusion_data(limit=limit))
