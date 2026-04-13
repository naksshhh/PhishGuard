"""
PhishGuard++ — Email Analysis Orchestrator (Tier 4)

3-branch cascade for email phishing detection:
    Branch 1: Header Forensics (instant, rule engine)
    Branch 2: RoBERTa + Cialdini (SOTA transformer)
    Branch 3: Gemini Suspicion-Primed (LLM fallback)

Mirrors the existing URL cascade logic (Tiers 1–3) — runs as a
parallel analysis pathway when the user is viewing email content.
"""

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Request/Response Schemas ─────────────────────────────────


class EmailAnalysisRequest(BaseModel):
    subject: str
    sender_display: str
    sender_email: str
    body_text: str
    body_html: Optional[str] = None
    headers: Optional[dict] = None
    reply_to: Optional[str] = None


class EmailAnalysisResponse(BaseModel):
    verdict: str  # PHISH / SAFE / SUSPICIOUS
    score: float
    branch: str  # header / roberta / gemini
    reason: str
    ai_generated_likelihood: Optional[float] = None
    persuasion_tactics: Optional[list] = None
    header_details: Optional[dict] = None


# ── Cascade Thresholds ───────────────────────────────────────

HEADER_FAIL_THRESHOLD = 0.6  # Branch 1: auto-PHISH if score >= this
ROBERTA_CONFIDENT_PHISH = 0.80  # Branch 2: confident phish
ROBERTA_CONFIDENT_SAFE = 0.20  # Branch 2: confident safe


# ── Orchestrator ─────────────────────────────────────────────


async def analyze_email(request: EmailAnalysisRequest) -> EmailAnalysisResponse:
    """
    Run the 3-branch email analysis cascade.

    Flow:
        1. Header Forensics → if FAIL, return PHISH immediately
        2. RoBERTa + Cialdini → if confident, return
        3. Gemini Suspicion-Primed → final arbiter
    """

    # ── Branch 1: Header Forensics ───────────────────────────
    from backend.header_forensics import analyze as run_header_forensics

    header_result = run_header_forensics(
        sender_display=request.sender_display,
        sender_email=request.sender_email,
        reply_to=request.reply_to,
        headers=request.headers,
    )

    logger.info(
        f"Branch 1 (Headers): {header_result.verdict} "
        f"(score={header_result.score:.4f})"
    )

    # Short-circuit on clear spoofing
    if header_result.verdict == "FAIL":
        failed_summary = "; ".join(header_result.failed_checks[:3])
        return EmailAnalysisResponse(
            verdict="PHISH",
            score=header_result.score,
            branch="header",
            reason=f"Email authentication failed: {failed_summary}",
            header_details=header_result.details,
        )

    # ── Branch 2: RoBERTa + Cialdini ────────────────────────
    roberta_score = 0.5
    roberta_verdict = "UNAVAILABLE"
    persuasion_tactics = []

    try:
        from src.features.cialdini_features import (
            extract_cialdini_features,
            get_dominant_tactics,
        )
        from src.models.roberta_email import roberta_inference

        cialdini = extract_cialdini_features(request.body_text)
        persuasion_tactics = get_dominant_tactics(request.body_text)

        # Run inference in a thread to not block the async loop
        roberta_score, roberta_verdict = await asyncio.to_thread(
            roberta_inference.predict,
            request.subject,
            request.body_text,
            cialdini,
        )

        logger.info(
            f"Branch 2 (RoBERTa): {roberta_verdict} "
            f"(score={roberta_score:.4f}, tactics={persuasion_tactics})"
        )

    except Exception as e:
        logger.warning(f"Branch 2 (RoBERTa) failed: {e}")

    # Confident RoBERTa verdict
    if roberta_score >= ROBERTA_CONFIDENT_PHISH:
        tactic_str = (
            f" Persuasion tactics detected: {', '.join(persuasion_tactics)}."
            if persuasion_tactics
            else ""
        )
        return EmailAnalysisResponse(
            verdict="PHISH",
            score=roberta_score,
            branch="roberta",
            reason=f"AI classifier detected phishing patterns with high confidence.{tactic_str}",
            persuasion_tactics=persuasion_tactics,
            header_details=header_result.details,
        )

    if roberta_score <= ROBERTA_CONFIDENT_SAFE and header_result.verdict == "PASS":
        return EmailAnalysisResponse(
            verdict="SAFE",
            score=roberta_score,
            branch="roberta",
            reason="Email passed header authentication and AI content analysis.",
            persuasion_tactics=persuasion_tactics,
            header_details=header_result.details,
        )

    # ── Branch 3: Gemini Suspicion-Primed ────────────────────
    logger.info(
        f"Branches 1+2 ambiguous (header={header_result.score:.2f}, "
        f"roberta={roberta_score:.2f}). Escalating to Gemini..."
    )

    try:
        from backend.gemini_email import analyze_email as gemini_analyze

        # Provide header context to Gemini for better analysis
        header_context = None
        if header_result.failed_checks:
            header_context = "\n".join(
                f"- {c}" for c in header_result.failed_checks
            )

        gemini_result = await asyncio.to_thread(
            gemini_analyze,
            subject=request.subject,
            sender_display=request.sender_display,
            sender_email=request.sender_email,
            body_text=request.body_text,
            reply_to=request.reply_to,
            header_forensics_summary=header_context,
        )

        logger.info(
            f"Branch 3 (Gemini): {gemini_result['verdict']} "
            f"(score={gemini_result['score']:.2f})"
        )

        # Merge persuasion tactics from Cialdini features and Gemini
        all_tactics = list(
            set(persuasion_tactics + gemini_result.get("persuasion_tactics", []))
        )

        return EmailAnalysisResponse(
            verdict=gemini_result["verdict"],
            score=gemini_result["score"],
            branch="gemini",
            reason=gemini_result["explanation"],
            ai_generated_likelihood=gemini_result.get(
                "ai_generated_likelihood"
            ),
            persuasion_tactics=all_tactics,
            header_details=header_result.details,
        )

    except Exception as e:
        logger.error(f"Branch 3 (Gemini) failed: {e}")

        # Fallback: use RoBERTa's ambiguous score
        verdict = "SUSPICIOUS" if roberta_score > 0.4 else "SAFE"
        return EmailAnalysisResponse(
            verdict=verdict,
            score=roberta_score,
            branch="roberta",
            reason=(
                f"Email analysis partially complete (Gemini unavailable). "
                f"Exercise caution — RoBERTa score: {roberta_score:.2f}."
            ),
            persuasion_tactics=persuasion_tactics,
            header_details=header_result.details,
        )
