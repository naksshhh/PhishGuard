"""
PhishGuard++ — Cloud Backend (FastAPI)
Tier 2 (Cloud Classifier) & Tier 3 (Gemini Multi-modal) Orchestration
"""

import logging
import os
import asyncio
import httpx
import pandas as pd
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

# Absolute imports
from src.explainability.shap_pipeline import get_phish_explanation, FEATURE_ORDER

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Gemini
_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL_ID = "gemini-2.5-flash-preview-04-17"

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
    screenshotBase64: Optional[str] = None
    features: Optional[dict] = None

class AnalysisResponse(BaseModel):
    verdict: str
    score: float
    tier: int
    reason: str

class ReportRequest(BaseModel):
    url: str
    reason: Optional[str] = ""
    
class TrustCheckResponse(BaseModel):
    found: bool
    report_count: int
    reasons: list
    verdict: str

# ── Tier 1.5: Safe Browsing ───────────────────────────────────
async def check_safe_browsing(url: str):
    """Hits Google Safe Browsing API to check for blacklisted URLs."""
    api_key = os.getenv("SAFE_BROWSING_API_KEY")
    if not api_key:
        return False, "API Key Missing"
    
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload = {
        "client": {"clientId": "phishguard-plus", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(endpoint, json=payload, timeout=2.0)
            data = resp.json()
            if "matches" in data and len(data["matches"]) > 0:
                threat_type = data["matches"][0].get("threatType", "MALICIOUS")
                return True, f"Google Safe Browsing: {threat_type}"
            return False, "Clean"
    except Exception as e:
        logger.warning(f"Safe Browsing API failed: {e}")
        return False, "API Timeout/Error"

# ── Tier 2: Cloud Classifier ──────────────────────────────────
async def run_tier2_analysis(features: dict):
    """Uses the Stage 1 LightGBM model for cloud-side inference."""
    from src.explainability.shap_pipeline import explainer
    
    if not explainer.model:
        return 0.5  # Fallback
        
    try:
        # Run in a thread to avoid blocking the async loop if the model is large
        def _predict():
            df = pd.DataFrame([features], columns=FEATURE_ORDER)
            # Ensure model returns probability
            if hasattr(explainer.model, "predict_proba"):
                return explainer.model.predict_proba(df)[0][1]
            return explainer.model.predict(df)[0]
            
        score = await asyncio.to_thread(_predict)
        return float(score)
    except Exception as e:
        logger.error(f"Tier 2 Inference failed: {e}")
        return 0.5

# ── Tier 3: Gemini Analysis ───────────────────────────────────
def run_tier3_gemini(url: str, html_excerpt: str, screenshot_base64: Optional[str] = None):
    import json, base64
    logger.info(f"Escalating to Gemini Tier 3 for {url}...")
    
    prompt = f"""Analyze the following URL and HTML excerpt for phishing indicators.
URL: {url}
HTML Excerpt:
{html_excerpt[:2000]}

Verify:
1. Brand impersonation (e.g., 'g00gle.com').
2. Credential harvesting forms (password/PII inputs to suspicious actions).
3. Urgency/threatening language.
4. If a screenshot image is provided, check if the logo/design matches the claimed brand.

Output ONLY valid JSON:
{{
  "verdict": "PHISH" or "SAFE",
  "score": 0.0 to 1.0,
  "reason": "Clear explanation."
}}"""
    
    parts = [genai_types.Part.from_text(text=prompt)]
    
    if screenshot_base64:
        raw_b64 = screenshot_base64.split(",")[1] if "," in screenshot_base64 else screenshot_base64
        parts.append(genai_types.Part.from_bytes(
            data=base64.b64decode(raw_b64),
            mime_type="image/jpeg"
        ))
    
    try:
        response = _genai_client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=parts,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        data = json.loads(response.text)
        return data.get("verdict", "SAFE").upper(), float(data.get("score", 0.1)), data.get("reason", "No reason.")
    except json.JSONDecodeError as e:
        logger.error(f"Gemini JSON Parse Error: {e}")
        return "ERROR", 0.5, "Tier 3 JSON parsing failed."
    except Exception as e:
        logger.error(f"Gemini Tier 3 failed: {e}")
        return "ERROR", 0.5, f"Tier 3 Gemini API error: {str(e)}"

@app.post("/analyze/cloud", response_model=AnalysisResponse)
async def analyze_cloud(request: AnalysisRequest):
    from src.models.inference_pipeline import orchestrator
    
    logger.info(f"Analyzing Cloud Tiers for: {request.url}")
    
    # 1. Run Tier 1.5 (Safe Browsing) and Tier 2 (LightGBM) in PARALLEL
    sb_task = check_safe_browsing(request.url)
    
    if request.features:
        lgbm_task = run_tier2_analysis(request.features)
        sb_result, t2_score = await asyncio.gather(sb_task, lgbm_task)
    else:
        sb_result, _ = await asyncio.gather(sb_task, asyncio.sleep(0))
        t2_score = 0.5

    is_blacklisted, sb_reason = sb_result

    # 2. Threshold check for Tier 1.5 (CRITICAL Hit)
    if is_blacklisted:
        return AnalysisResponse(verdict="PHISH", score=1.0, tier=1, reason=f"CRITICAL: {sb_reason}")

    # 3. Generate SHAP Explanation
    explanation = ""
    if request.features and t2_score >= 0.5:
        try:
            explanation = get_phish_explanation(request.features)
        except Exception:
            explanation = "Structural anomalies detected."

    # 4. Multimodal Fusion (Tabular + URL + HTML + Vision)
    fusion_data = await orchestrator.fuse_predictions(
        tabular_score=t2_score,
        url=request.url,
        html=request.htmlExcerpt,
        screenshot=request.screenshotBase64
    )
    
    fused_score = fusion_data["fused_score"]
    
    # 5. Fast-Path for highly confident local models
    if fused_score > 0.8:
        return AnalysisResponse(
            verdict="PHISH", 
            score=fused_score, 
            tier=3, 
            reason=f"Fusion Confidence [{fused_score:.2f}]: {explanation or 'High Multimodal Risk.'}"
        )
    elif fused_score < 0.2:
        return AnalysisResponse(
            verdict="SAFE", 
            score=fused_score, 
            tier=3, 
            reason="Multi-tier signals appear safe."
        )
        
    # 6. Escalate to Gemini (Tier 4 fallback) if local models are ambiguous
    logger.info(f"Fusion score ambiguous ({fused_score:.2f}), escalating to Gemini...")
    verdict, g_score, vision_reason = run_tier3_gemini(request.url, request.htmlExcerpt, request.screenshotBase64)
    final_reason = f"{vision_reason}\n\nTechnical Signals:\n{explanation}" if explanation else vision_reason
    return AnalysisResponse(verdict=verdict, score=g_score, tier=4, reason=final_reason)

@app.on_event("startup")
def startup_event():
    # We dynamically import so it doesn't break if run from root.
    from . import firebase_db
    firebase_db.init_firebase()

@app.post("/community/report")
async def report_url(request: ReportRequest):
    from . import firebase_db
    success = firebase_db.report_malicious_url(request.url, request.reason)
    if success:
        return {"status": "success", "message": "Report logged."}
    else:
        raise HTTPException(status_code=500, detail="Failed to log report to DB.")

@app.get("/community/check", response_model=TrustCheckResponse)
async def check_url(url: str):
    from . import firebase_db
    result = firebase_db.get_community_trust(url)
    
    if "error" in result:
        # Return graceful failure instead of 500 so extension keeps working
        return TrustCheckResponse(found=False, report_count=0, reasons=[], verdict="ERROR")
        
    return TrustCheckResponse(
        found=result.get("found", False),
        report_count=result.get("report_count", 0),
        reasons=result.get("reasons", []),
        verdict=result.get("verdict", "PENDING")
    )

# ── Tier 4: Email Analysis ────────────────────────────────────

class EmailAnalysisRequest(BaseModel):
    subject: str
    sender_display: str
    sender_email: str
    body_text: str
    body_html: Optional[str] = None
    headers: Optional[dict] = None
    reply_to: Optional[str] = None

class EmailAnalysisResponse(BaseModel):
    verdict: str
    score: float
    branch: str
    reason: str
    ai_generated_likelihood: Optional[float] = None
    persuasion_tactics: Optional[list] = None
    header_details: Optional[dict] = None

class SenderReportRequest(BaseModel):
    domain: str
    reason: Optional[str] = ""

class SenderTrustResponse(BaseModel):
    found: bool
    report_count: int
    reasons: list
    verdict: str


@app.post("/analyze/email", response_model=EmailAnalysisResponse)
async def analyze_email_endpoint(request: EmailAnalysisRequest):
    """
    Tier 4: Email Phishing Detection via 3-branch cascade.
    Branch 1: Header Forensics (instant)
    Branch 2: RoBERTa + Cialdini persuasion features
    Branch 3: Gemini suspicion-primed analysis
    """
    from backend.email_analyzer import analyze_email as run_email_analysis
    from backend.email_analyzer import EmailAnalysisRequest as InternalRequest

    logger.info(f"Email Analysis: '{request.subject}' from {request.sender_email}")

    internal_req = InternalRequest(
        subject=request.subject,
        sender_display=request.sender_display,
        sender_email=request.sender_email,
        body_text=request.body_text,
        body_html=request.body_html,
        headers=request.headers,
        reply_to=request.reply_to,
    )

    result = await run_email_analysis(internal_req)

    return EmailAnalysisResponse(
        verdict=result.verdict,
        score=result.score,
        branch=result.branch,
        reason=result.reason,
        ai_generated_likelihood=result.ai_generated_likelihood,
        persuasion_tactics=result.persuasion_tactics,
        header_details=result.header_details,
    )


@app.post("/community/report-sender")
async def report_sender(request: SenderReportRequest):
    """Report a malicious sender domain to the community."""
    from . import firebase_db
    success = firebase_db.report_malicious_sender(request.domain, request.reason)
    if success:
        return {"status": "success", "message": "Sender domain reported."}
    else:
        raise HTTPException(status_code=500, detail="Failed to log sender report.")


@app.get("/community/check-sender", response_model=SenderTrustResponse)
async def check_sender(domain: str):
    """Check community trust for a sender domain."""
    from . import firebase_db
    result = firebase_db.get_sender_trust(domain)

    if "error" in result:
        return SenderTrustResponse(found=False, report_count=0, reasons=[], verdict="ERROR")

    return SenderTrustResponse(
        found=result.get("found", False),
        report_count=result.get("report_count", 0),
        reasons=result.get("reasons", []),
        verdict=result.get("verdict", "PENDING")
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

