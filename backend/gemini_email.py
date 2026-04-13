"""
PhishGuard++ — Gemini Suspicion-Primed Email Analysis (Branch 3)

Uses Google Gemini 1.5 Flash with "suspicion priming" — leading with
"Is there anything suspicious about this email?" rather than
"classify this email" — a technique shown to achieve 100% detection
rate without increasing false positives (arXiv 2024).

The ai_generated_likelihood field is a novel contribution — no
existing deployed email security tool exposes this score to end users.
"""

import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Reuse the same Gemini client pattern from main.py
_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL_ID = "gemini-2.5-flash-preview-04-17"

# ── Suspicion-Primed Prompt ──────────────────────────────────

SUSPICION_PROMPT = """You are an expert email security analyst. Your job is to protect users from phishing, social engineering, and AI-generated spear phishing attacks.

**Is there anything suspicious about this email?**

Analyze the following email carefully:

FROM: {sender_display} <{sender_email}>
SUBJECT: {subject}
REPLY-TO: {reply_to}

--- EMAIL BODY ---
{body}
--- END ---

{header_context}

Check for ALL of the following:
1. Brand impersonation — does the sender claim to represent a company their domain doesn't belong to?
2. Credential harvesting — does the email ask for passwords, SSN, credit card, or other PII?
3. Urgency/threatening language — account suspension, legal action, deadline pressure?
4. AI-generated content — is the writing style unnaturally perfect, lacks personal context, or follows a formulaic persuasion pattern?
5. Suspicious links — any URLs that don't match the claimed sender's domain?
6. Social engineering tactics — authority claims, reciprocity, scarcity, flattery?

Output ONLY valid JSON with this exact structure:
{{
    "verdict": "PHISH" or "SAFE",
    "score": 0.0 to 1.0,
    "impersonated_brand": "BrandName" or null,
    "urgency_language": true or false,
    "suspicious_links": ["url1", "url2"] or [],
    "credential_request": true or false,
    "ai_generated_likelihood": 0.0 to 1.0,
    "persuasion_tactics": ["urgency", "authority"] or [],
    "explanation": "One clear sentence explaining the verdict to a non-technical user."
}}"""


def analyze_email(
    subject: str,
    sender_display: str,
    sender_email: str,
    body_text: str,
    reply_to: Optional[str] = None,
    header_forensics_summary: Optional[str] = None,
) -> dict:
    """
    Run Gemini suspicion-primed analysis on an email.

    Args:
        subject: Email subject line
        sender_display: Sender display name
        sender_email: Sender email address
        body_text: Plain text email body
        reply_to: Reply-To address if different from sender
        header_forensics_summary: Summary of Branch 1 findings to prime Gemini

    Returns:
        Dictionary with verdict, score, and all analysis fields
    """
    logger.info(f"Gemini Email Analysis: '{subject}' from {sender_email}")

    # Build header context if Branch 1 found issues
    header_context = ""
    if header_forensics_summary:
        header_context = (
            f"SECURITY NOTE: Pre-analysis detected these header issues:\n"
            f"{header_forensics_summary}\n"
            f"Factor these into your analysis."
        )

    prompt = SUSPICION_PROMPT.format(
        sender_display=sender_display or "(none)",
        sender_email=sender_email or "(unknown)",
        subject=subject or "(no subject)",
        reply_to=reply_to or sender_email or "(same as sender)",
        body=body_text[:3000] if body_text else "(empty body)",
        header_context=header_context,
    )

    try:
        response = _genai_client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=[genai_types.Part.from_text(text=prompt)],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        data = json.loads(response.text)

        # Normalize and validate
        result = {
            "verdict": str(data.get("verdict", "SAFE")).upper(),
            "score": float(data.get("score", 0.1)),
            "impersonated_brand": data.get("impersonated_brand"),
            "urgency_language": bool(data.get("urgency_language", False)),
            "suspicious_links": list(data.get("suspicious_links", [])),
            "credential_request": bool(data.get("credential_request", False)),
            "ai_generated_likelihood": float(
                data.get("ai_generated_likelihood", 0.0)
            ),
            "persuasion_tactics": list(data.get("persuasion_tactics", [])),
            "explanation": str(
                data.get("explanation", "Analysis complete.")
            ),
        }

        # Ensure verdict is valid
        if result["verdict"] not in ("PHISH", "SAFE"):
            result["verdict"] = "SAFE" if result["score"] < 0.5 else "PHISH"

        logger.info(
            f"Gemini Email Verdict: {result['verdict']} "
            f"(score={result['score']:.2f}, "
            f"ai_gen={result['ai_generated_likelihood']:.2f})"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Gemini email JSON parse error: {e}")
        return _error_response("Gemini returned invalid JSON.")

    except Exception as e:
        logger.error(f"Gemini email analysis failed: {e}")
        return _error_response(f"Gemini API error: {str(e)}")


def _error_response(reason: str) -> dict:
    """Return a safe fallback response when Gemini fails."""
    return {
        "verdict": "ERROR",
        "score": 0.5,
        "impersonated_brand": None,
        "urgency_language": False,
        "suspicious_links": [],
        "credential_request": False,
        "ai_generated_likelihood": 0.0,
        "persuasion_tactics": [],
        "explanation": reason,
    }
