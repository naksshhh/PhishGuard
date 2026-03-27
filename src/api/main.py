"""
PhishGuard++ — Cloud Backend (FastAPI)
Tier 2 (Cloud Classifier) & Tier 3 (Gemini Multi-modal) Orchestration
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title="PhishGuard++ Cloud Backend")

# Allow Chrome extension and localhost to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    url: str
    htmlExcerpt: str
    features: Optional[dict] = None

class AnalysisResponse(BaseModel):
    verdict: str
    score: float
    tier: int
    reason: str

# ── Tier 2: Cloud Classifier ──────────────────────────────────
# (Expects a trained model like XGBoost or a PyTorch NN)
# For Phase 1 demo, we use a placeholder logic or the VAE model.
def run_tier2_analysis(url: str, html_excerpt: str):
    logger.info(f"Running Tier 2 Analysis for {url}...")
    # Placeholder: In production, load your cloud_model.pkl here
    score = 0.65 # Arbitrary suspicious score
    return score

# ── Tier 3: Gemini Analysis ───────────────────────────────────
def run_tier3_gemini(url: str, html_excerpt: str):
    logger.info(f"Escalating to Tier 3 (Gemini) for {url}...")
    
    prompt = f"""
    Analyze the following URL and HTML excerpt for phishing indicators.
    URL: {url}
    HTML Excerpt:
    {html_excerpt}
    
    Verify:
    1. Brand impersonation (e.g., 'g00gle.com').
    2. Credential harvesting forms (password/PII inputs to suspicious actions).
    3. Urgency/Threatening language.
    
    Return a JSON response with:
    "verdict": "PHISH" or "SAFE",
    "score": 0.0 - 1.0,
    "reason": "Clear explanation of the threat"
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        # Simplified parsing (in production, use structured output/json mode)
        text = response.text
        if "PHISH" in text.upper():
            return "PHISH", 0.95, text[:200]
        return "SAFE", 0.1, "Gemini verified site as legitimate."
    except Exception as e:
        logger.error(f"Gemini Tier 3 failed: {e}")
        return "ERROR", 0.5, "Inconclusive (Gemini unavailable)."

@app.post("/analyze/cloud", response_model=AnalysisResponse)
async def analyze_cloud(request: AnalysisRequest):
    # 1. Run Tier 2
    t2_score = run_tier2_analysis(request.url, request.htmlExcerpt)
    
    # 2. Threshold check for Tier 3 escalation
    if t2_score > 0.8:
        return AnalysisResponse(verdict="PHISH", score=t2_score, tier=2, reason="High structural risk.")
    
    # 3. Escalate to Tier 3 if ambiguous (e.g., score around 0.5-0.7)
    if t2_score >= 0.5:
        verdict, score, reason = run_tier3_gemini(request.url, request.htmlExcerpt)
        return AnalysisResponse(verdict=verdict, score=score, tier=3, reason=reason)
    
    return AnalysisResponse(verdict="SAFE", score=t2_score, tier=2, reason="Structural signals appear safe.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
