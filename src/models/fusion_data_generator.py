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

# ── Branch Score Collectors ──────────────────────────────────

async def get_tabular_score(features: dict):
    """Tier 2 Cloud (LightGBM)"""
    # This usually comes from backend logic
    return 0.5 # Placeholder

async def get_semantic_url_score(url: str):
    """Tier 3 PhishBERT"""
    return 0.5 # Placeholder

async def get_semantic_html_score(html_excerpt: str):
    """Tier 3 CodeBERT"""
    if len(html_excerpt) < 100:
        return 0.5, 0.0 # score, mask
    return 0.5, 1.0

async def get_visual_score(image_path: Path):
    """Tier 3 EfficientNet"""
    if not image_path.exists():
        return 0.5, 0.0 # score, mask
    return 0.5, 1.0

# ─────────────────────────────────────────────────────────────

async def generate_fusion_data():
    """Run all branches on the validation set to create the meta-dataset."""
    val_path = DATA_DIR / "val_features.csv"
    if not val_path.exists():
        logger.error("Validation data not found.")
        return

    df = pd.read_csv(val_path)
    # Target only rows with HTML filenames for now
    df = df.dropna(subset=["html_filename"])
    
    logger.info(f"Generating fusion data for {len(df)} validation samples...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Tabular Branch
        tabular_score = await get_tabular_score(row.to_dict())
        
        # 2. PhishBERT URL Branch
        url_score = await get_semantic_url_score(row["url"])
        
        # 3. CodeBERT HTML Branch (With Gating)
        html_file = BASE_DIR / "datasets" / "Mendeley phishing dataset" / row["html_filename"]
        html_text = ""
        if html_file.exists():
            with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                html_text = f.read(2048)
        
        html_score, html_mask = await get_semantic_html_score(html_text)
        
        # 4. EfficientNet Visual Branch (With Gating)
        img_path = BASE_DIR / "datasets" / "screenshots" / row["html_filename"].replace(".html", ".jpg")
        visual_score, visual_mask = await get_visual_score(img_path)
        
        results.append({
            "tabular_score": tabular_score,
            "url_score": url_score,
            "html_score": html_score,
            "html_mask": html_mask,
            "visual_score": visual_score,
            "visual_mask": visual_mask,
            "label": row["label"]
        })

    fusion_df = pd.DataFrame(results)
    output_path = DATA_DIR / "fusion_train_v2.csv"
    fusion_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved fusion meta-dataset to {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_fusion_data())
