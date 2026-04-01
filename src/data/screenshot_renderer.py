import asyncio
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
HTML_DIR = BASE_DIR / "datasets" / "Mendeley phishing dataset"
OUTPUT_DIR = BASE_DIR / "datasets" / "screenshots"

# Max parallel workers (adjust based on CPU/RAM)
MAX_WORKERS = 8
# Target sample size for SOTA training (81k is overkill for a laptop)
SAMPLE_LIMIT = 80000

async def render_task(browser, html_file, semaphore, pbar):
    """Worker task to render a single HTML file."""
    async with semaphore:
        screenshot_name = html_file.stem + ".jpg"
        output_path = OUTPUT_DIR / screenshot_name
        
        if output_path.exists():
            pbar.update(1)
            return True
            
        page = await browser.new_page(viewport={'width': 1280, 'height': 720})
        try:
            url = f"file://{html_file.resolve()}"
            # Fast-path: wait until domcontentloaded instead of networkidle for local files
            await page.goto(url, timeout=3000, wait_until="domcontentloaded")
            await page.screenshot(path=str(output_path), type='jpeg', quality=80)
            return True
        except Exception:
            return False
        finally:
            await page.close()
            pbar.update(1)

async def batch_render():
    """Render a representative sample of HTML files in parallel."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    html_files = list(HTML_DIR.glob("*.html"))
    # Shuffle or sample to get a manageable dataset
    import random
    random.seed(42)
    if len(html_files) > SAMPLE_LIMIT:
        logger.info(f"Sampling {SAMPLE_LIMIT} files from {len(html_files)} to avoid 200+ hour runtime.")
        html_files = random.sample(html_files, SAMPLE_LIMIT)
    
    logger.info(f"Starting parallel rendering with {MAX_WORKERS} workers...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        semaphore = asyncio.Semaphore(MAX_WORKERS)
        
        with tqdm(total=len(html_files), desc="Rendering Screenshots") as pbar:
            tasks = [render_task(browser, f, semaphore, pbar) for f in html_files]
            results = await asyncio.gather(*tasks)
            
        await browser.close()
        
    success_count = sum(1 for r in results if r)
    logger.info(f"✅ Rendering complete. Created {success_count} screenshots in {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(batch_render())

if __name__ == "__main__":
    asyncio.run(batch_render())
