"""
PhishGuard++ — Cialdini Persuasion Principle Feature Extractor

Extracts a 6-dimensional feature vector quantifying the presence of
Robert Cialdini's six persuasion principles in email body text.

Based on the 2024 arXiv paper achieving 91% F1 on LLM-generated spear
phishing by using prompted contextual document vectors incorporating
these persuasion patterns.

Each dimension is a normalized score [0.0, 1.0] representing the
intensity of that persuasion principle in the text.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Principle 1: Urgency ─────────────────────────────────────
URGENCY_PATTERNS = [
    r"\burgent(ly)?\b",
    r"\bimmediatel(y|e)\b",
    r"\bexpir(e|es|ed|ing)\b",
    r"\bwithin\s+\d+\s+(hour|day|minute)",
    r"\bact\s+now\b",
    r"\blast\s+chance\b",
    r"\bdo\s+not\s+delay\b",
    r"\basap\b",
    r"\btime[\s-]sensitive\b",
    r"\bdeadline\b",
    r"\brequir(e|es|ed)\s+(immediate|prompt)",
    r"\btoday\s+only\b",
    r"\bhurry\b",
    r"\bexpiration\b",
    r"\bsuspend(ed)?\b",
    r"\bterminat(e|ed|ion)\b",
    r"\baction\s+required\b",
    r"\bfinal\s+notice\b",
    r"\bresponse\s+needed\b",
    r"\bbefore\s+it'?s?\s+too\s+late\b",
]

# ── Principle 2: Scarcity ────────────────────────────────────
SCARCITY_PATTERNS = [
    r"\blimited\s+(time|offer|availability|spots?)\b",
    r"\bonly\s+\d+\s+(left|remaining|available)\b",
    r"\bexclusive\s+(offer|deal|access)\b",
    r"\bwhile\s+supplies?\s+last\b",
    r"\brare\s+opportunity\b",
    r"\bone[\s-]time\s+(offer|deal)\b",
    r"\bfirst[\s-]come\b",
    r"\bselling\s+fast\b",
    r"\bfew\s+(spots?|seats?|items?)\s+(left|remaining)\b",
    r"\bdon'?t\s+miss\s+(out|this)\b",
]

# ── Principle 3: Authority ───────────────────────────────────
AUTHORITY_PATTERNS = [
    r"\bceo\b",
    r"\bcfo\b",
    r"\bcto\b",
    r"\bdirector\b",
    r"\bmanaging\s+director\b",
    r"\bvice\s+president\b",
    r"\bpresident\b",
    r"\bchief\s+(executive|financial|technology|information)\b",
    r"\bit[\s-]department\b",
    r"\bsecurity\s+team\b",
    r"\bcompliance\s+(team|department|officer)\b",
    r"\blegal\s+(team|department)\b",
    r"\bhuman\s+resources?\b",
    r"\bmanagement\b",
    r"\bboard\s+of\s+directors?\b",
    r"\bgovernment\b",
    r"\bfederal\b",
    r"\binternal\s+revenue\b",
    r"\birs\b",
    r"\bofficial\b",
    r"\bauthoriz(e|ed|ation)\b",
    r"\bverifi(ed|cation)\b",
]

# ── Principle 4: Social Proof ────────────────────────────────
SOCIAL_PROOF_PATTERNS = [
    r"\beveryone\b",
    r"\bother\s+(users?|customers?|employees?|colleagues?)\b",
    r"\bmany\s+(people|users?|customers?)\b",
    r"\bmost\s+(people|users?|employees?)\b",
    r"\bjoin\s+\d+",
    r"\btrusted\s+by\b",
    r"\bmillions?\s+(of\s+)?(users?|customers?|people)\b",
    r"\bpopular\b",
    r"\brecommended\s+by\b",
    r"\bco[\s-]?workers?\b",
    r"\bteam\s+members?\b",
    r"\byour\s+colleagues?\b",
]

# ── Principle 5: Liking ──────────────────────────────────────
LIKING_PATTERNS = [
    r"\bdear\s+(friend|customer|member|user|valued)\b",
    r"\bvalued\s+(customer|member|client)\b",
    r"\bhello\s+friend\b",
    r"\bwe\s+appreciate\b",
    r"\bthank\s+you\s+for\s+(your|being)\b",
    r"\bcongratulations\b",
    r"\byou'?ve\s+been\s+selected\b",
    r"\bspecial(ly)?\s+(for\s+you|chosen|selected)\b",
    r"\bflattering\b",
    r"\bwe\s+value\b",
    r"\bloyalt?y\b",
    r"\bpersonaliz(e|ed)\b",
    r"\bjust\s+for\s+you\b",
]

# ── Principle 6: Reciprocity ────────────────────────────────
RECIPROCITY_PATTERNS = [
    r"\bfree\s+(gift|trial|offer|access|account)\b",
    r"\bcomplimentary\b",
    r"\bno\s+(cost|charge|obligation)\b",
    r"\bgift\s+(card|certificate|for\s+you)\b",
    r"\bbonus\b",
    r"\breward\b",
    r"\bwe'?ve\s+already\b",
    r"\bas\s+a\s+thank\s+you\b",
    r"\bin\s+return\b",
    r"\bclaim\s+your\b",
    r"\brefund\b",
    r"\bcashback\b",
    r"\bcredit\s+to\s+your\b",
]

# All 6 principles collected
PRINCIPLE_NAMES = [
    "urgency",
    "scarcity",
    "authority",
    "social_proof",
    "liking",
    "reciprocity",
]

ALL_PATTERNS = [
    URGENCY_PATTERNS,
    SCARCITY_PATTERNS,
    AUTHORITY_PATTERNS,
    SOCIAL_PROOF_PATTERNS,
    LIKING_PATTERNS,
    RECIPROCITY_PATTERNS,
]


def _count_matches(text: str, patterns: List[str]) -> int:
    """Count how many distinct pattern matches are found in the text."""
    count = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count


def extract_cialdini_features(body_text: str) -> List[float]:
    """
    Extract a 6-dimensional Cialdini persuasion feature vector.

    Each dimension is normalized to [0.0, 1.0] by dividing the match
    count by the total number of patterns for that principle.

    Args:
        body_text: Raw email body text (no HTML tags)

    Returns:
        List of 6 floats: [urgency, scarcity, authority, social_proof, liking, reciprocity]
    """
    if not body_text or not body_text.strip():
        return [0.0] * 6

    # Normalize whitespace for more reliable matching
    text = re.sub(r"\s+", " ", body_text.lower().strip())

    features = []
    for patterns in ALL_PATTERNS:
        count = _count_matches(text, patterns)
        # Normalize: fraction of patterns that matched
        normalized = min(count / max(len(patterns), 1), 1.0)
        features.append(round(normalized, 4))

    return features


def extract_cialdini_dict(body_text: str) -> dict:
    """
    Same as extract_cialdini_features but returns a named dictionary.
    Useful for SHAP explanations and logging.
    """
    values = extract_cialdini_features(body_text)
    return dict(zip(PRINCIPLE_NAMES, values))


def get_dominant_tactics(body_text: str, threshold: float = 0.15) -> List[str]:
    """
    Return human-readable names of persuasion tactics exceeding
    the given threshold. Used for the Gemini response and UI display.
    """
    features = extract_cialdini_dict(body_text)
    return [name for name, score in features.items() if score >= threshold]
