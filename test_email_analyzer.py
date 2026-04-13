"""
PhishGuard++ — Email Analysis Integration Tests

Tests all 3 branches of the Tier 4 email pipeline:
    1. Header Forensics (rule engine)
    2. RoBERTa + Cialdini (if model available)
    3. Gemini Suspicion-Primed (if API key available)
    4. Firebase sender domain community
    5. Full cascade orchestration
"""

import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ═══════════════════════════════════════════════════════════════
# Branch 1: Header Forensics
# ═══════════════════════════════════════════════════════════════

class TestHeaderForensics:

    def test_clean_headers_pass(self):
        """Legitimate email with valid auth should PASS."""
        from backend.header_forensics import analyze

        result = analyze(
            sender_display="John Smith",
            sender_email="john@company.com",
            reply_to="john@company.com",
            headers={
                "authentication-results": "mx.google.com; spf=pass; dkim=pass; dmarc=pass",
            },
        )
        assert result.verdict == "PASS"
        assert result.score < 0.3
        assert len(result.failed_checks) == 0

    def test_spf_dkim_fail(self):
        """SPF and DKIM failure should produce high score."""
        from backend.header_forensics import analyze

        result = analyze(
            sender_display="Security Team",
            sender_email="alert@evil-domain.com",
            headers={
                "authentication-results": "mx.google.com; spf=fail; dkim=fail; dmarc=fail",
            },
        )
        assert result.verdict in ("FAIL", "WARN")
        assert result.score >= 0.3
        assert any("SPF" in f for f in result.failed_checks)
        assert any("DKIM" in f for f in result.failed_checks)

    def test_brand_impersonation(self):
        """Display name claiming to be PayPal but domain is different."""
        from backend.header_forensics import analyze

        result = analyze(
            sender_display="PayPal Security",
            sender_email="alert@totally-legit-123.com",
            headers={},
        )
        assert result.score > 0.0
        assert any("paypal" in f.lower() for f in result.failed_checks)

    def test_reply_to_mismatch(self):
        """Reply-To pointing to different domain."""
        from backend.header_forensics import analyze

        result = analyze(
            sender_display="HR Department",
            sender_email="hr@company.com",
            reply_to="hr-real@attacker.com",
            headers={
                "authentication-results": "spf=pass; dkim=pass; dmarc=pass",
            },
        )
        assert any("Reply-To" in f for f in result.failed_checks)

    def test_bulk_mailer_detection(self):
        """Known bulk mailing tool in X-Mailer."""
        from backend.header_forensics import analyze

        result = analyze(
            sender_display="Offers",
            sender_email="offers@spam.com",
            headers={
                "x-mailer": "Mass Mailer Pro 3.0",
            },
        )
        assert any("bulk" in f.lower() for f in result.failed_checks)

    def test_empty_headers_graceful(self):
        """Empty/missing headers should not crash."""
        from backend.header_forensics import analyze

        result = analyze(
            sender_display="",
            sender_email="user@example.com",
            headers=None,
        )
        assert result.verdict in ("PASS", "WARN")


# ═══════════════════════════════════════════════════════════════
# Cialdini Feature Extraction
# ═══════════════════════════════════════════════════════════════

class TestCialdiniFeatures:

    def test_urgency_detection(self):
        """Should detect urgency language."""
        from src.features.cialdini_features import extract_cialdini_dict

        features = extract_cialdini_dict(
            "URGENT: Your account will be suspended immediately! "
            "Act now before it's too late. This is your final notice."
        )
        assert features["urgency"] > 0.1

    def test_authority_detection(self):
        """Should detect authority claims."""
        from src.features.cialdini_features import extract_cialdini_dict

        features = extract_cialdini_dict(
            "This is from the CEO. The IT department requires your "
            "immediate compliance. This has been authorized by management."
        )
        assert features["authority"] > 0.1

    def test_scarcity_detection(self):
        """Should detect scarcity tactics."""
        from src.features.cialdini_features import extract_cialdini_dict

        features = extract_cialdini_dict(
            "Limited time offer! Only 3 spots left. "
            "This exclusive deal won't last. Don't miss out!"
        )
        assert features["scarcity"] > 0.1

    def test_clean_email_low_scores(self):
        """Normal business email should have low persuasion scores."""
        from src.features.cialdini_features import extract_cialdini_dict

        features = extract_cialdini_dict(
            "Hi team, please find the Q3 report attached. "
            "Let me know if you have questions. Best, Alice"
        )
        total = sum(features.values())
        assert total < 0.3  # Very low persuasion

    def test_empty_text(self):
        """Empty text should return all zeros."""
        from src.features.cialdini_features import extract_cialdini_features

        features = extract_cialdini_features("")
        assert features == [0.0] * 6

    def test_feature_dimensions(self):
        """Should always return exactly 6 features."""
        from src.features.cialdini_features import extract_cialdini_features

        features = extract_cialdini_features("Any text here")
        assert len(features) == 6
        assert all(0.0 <= f <= 1.0 for f in features)

    def test_dominant_tactics(self):
        """Should return names of tactics above threshold."""
        from src.features.cialdini_features import get_dominant_tactics

        tactics = get_dominant_tactics(
            "URGENT! Act immediately! This expires today! "
            "Only 2 spots remaining, limited availability!",
            threshold=0.1,
        )
        assert "urgency" in tactics


# ═══════════════════════════════════════════════════════════════
# Full Cascade (End-to-End)
# ═══════════════════════════════════════════════════════════════

class TestEmailCascade:

    @pytest.mark.asyncio
    async def test_header_short_circuit(self):
        """Cascade should short-circuit on clear header spoofing."""
        from backend.email_analyzer import analyze_email, EmailAnalysisRequest

        request = EmailAnalysisRequest(
            subject="Verify Your Account",
            sender_display="PayPal",
            sender_email="security@ph1shing-domain.xyz",
            body_text="Click here to verify your PayPal account.",
            headers={
                "authentication-results": "spf=fail; dkim=fail; dmarc=fail",
            },
        )
        result = await analyze_email(request)
        assert result.verdict == "PHISH"
        assert result.branch == "header"

    @pytest.mark.asyncio
    async def test_safe_email_passes(self):
        """Clean email with valid headers should pass."""
        from backend.email_analyzer import analyze_email, EmailAnalysisRequest

        request = EmailAnalysisRequest(
            subject="Q3 Report",
            sender_display="Alice Johnson",
            sender_email="alice@company.com",
            body_text=(
                "Hi team, the quarterly report is ready for review. "
                "Please find it in the shared drive. Best regards, Alice."
            ),
            headers={
                "authentication-results": "spf=pass; dkim=pass; dmarc=pass",
            },
        )
        result = await analyze_email(request)
        # Should not be classified as PHISH
        assert result.verdict in ("SAFE", "SUSPICIOUS")


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

class TestHelpers:

    def test_extract_domain(self):
        from backend.header_forensics import _extract_domain

        assert _extract_domain("user@example.com") == "example.com"
        assert _extract_domain("User <user@sub.example.co.uk>") == "sub.example.co.uk"
        assert _extract_domain("noatsign") == ""

    def test_extract_display_name(self):
        from backend.header_forensics import _extract_display_name

        assert _extract_display_name('"PayPal" <noreply@paypal.com>') == "PayPal"
        assert _extract_display_name("John Smith <john@co.com>") == "John Smith"
        assert _extract_display_name("just-email@test.com") == ""

    def test_parse_auth_results(self):
        from backend.header_forensics import _parse_auth_results

        parsed = _parse_auth_results(
            "mx.google.com; spf=pass (google.com); dkim=pass; dmarc=pass"
        )
        assert parsed["spf"] == "pass"
        assert parsed["dkim"] == "pass"
        assert parsed["dmarc"] == "pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
