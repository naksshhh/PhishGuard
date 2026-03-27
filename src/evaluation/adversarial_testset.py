"""
PhishGuard++ — Adversarial Test Set Generator
Creates 200+ adversarial URLs designed to evade naive classifiers,
used for robustness evaluation.
"""

import logging
import random
import string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Known Brand Targets ────────────────────────────────────────
BRANDS = [
    "paypal", "google", "apple", "amazon", "microsoft",
    "netflix", "facebook", "instagram", "linkedin", "dropbox",
    "chase", "bankofamerica", "wellsfargo", "citibank", "usaa",
]

# ── Homoglyph Substitutions ───────────────────────────────────
HOMOGLYPHS = {
    "a": ["à", "á", "â", "ã", "ä", "å", "ą", "α"],
    "e": ["è", "é", "ê", "ë", "ę", "ε"],
    "i": ["ì", "í", "î", "ï", "1", "l", "ι"],
    "o": ["ò", "ó", "ô", "õ", "ö", "0", "ο"],
    "l": ["1", "I", "ι"],
    "g": ["q", "ɡ"],
    "n": ["η", "ñ"],
    "s": ["$", "ş", "ś"],
}

# ── Attack Strategies ──────────────────────────────────────────

def homoglyph_attack(brand: str) -> list[str]:
    """Replace characters with visually similar alternatives."""
    results = []
    for i, ch in enumerate(brand):
        if ch in HOMOGLYPHS:
            for sub in HOMOGLYPHS[ch][:2]:
                results.append(brand[:i] + sub + brand[i+1:])
    return results


def subdomain_attack(brand: str) -> list[str]:
    """Place brand in subdomain of an arbitrary domain."""
    suffixes = [".com", ".net", ".org", ".xyz", ".info"]
    return [
        f"https://{brand}.login.{''.join(random.choices(string.ascii_lowercase, k=6))}{s}"
        for s in suffixes
    ]


def path_injection_attack(brand: str) -> list[str]:
    """Place brand in URL path, not domain."""
    domains = ["securesite.xyz", "login-verify.com", "account-update.net"]
    return [f"https://{d}/{brand}/account/verify" for d in domains]


def typosquatting_attack(brand: str) -> list[str]:
    """Insert, delete, or swap characters."""
    attacks = []
    # Insertion
    pos = random.randint(1, len(brand) - 1)
    attacks.append(brand[:pos] + random.choice(string.ascii_lowercase) + brand[pos:])
    # Deletion
    pos = random.randint(0, len(brand) - 1)
    attacks.append(brand[:pos] + brand[pos+1:])
    # Swap
    if len(brand) > 2:
        pos = random.randint(0, len(brand) - 3)
        attacks.append(brand[:pos] + brand[pos+1] + brand[pos] + brand[pos+2:])
    return [f"https://{a}.com" for a in attacks]


def url_padding_attack(brand: str) -> list[str]:
    """Pad URL with many characters to obscure the real domain."""
    padding = "secure-login-verify-account-update"
    return [
        f"https://{brand}-{padding}.malicious.com",
        f"https://{padding}-{brand}.evil.net",
    ]


def punycode_attack(brand: str) -> list[str]:
    """Use xn-- punycode encoding."""
    return [f"https://xn--{brand[:3]}p1ai.com"]


def generate_adversarial_testset() -> list[dict]:
    """Generate a comprehensive adversarial URL test set."""
    logger.info("Generating adversarial URL test set...")
    test_set = []

    for brand in BRANDS:
        for url in homoglyph_attack(brand):
            test_set.append({"url": f"https://{url}.com", "attack_type": "homoglyph", "brand": brand, "label": 1})

        for url in subdomain_attack(brand):
            test_set.append({"url": url, "attack_type": "subdomain", "brand": brand, "label": 1})

        for url in path_injection_attack(brand):
            test_set.append({"url": url, "attack_type": "path_injection", "brand": brand, "label": 1})

        for url in typosquatting_attack(brand):
            test_set.append({"url": url, "attack_type": "typosquatting", "brand": brand, "label": 1})

        for url in url_padding_attack(brand):
            test_set.append({"url": url, "attack_type": "url_padding", "brand": brand, "label": 1})

        for url in punycode_attack(brand):
            test_set.append({"url": url, "attack_type": "punycode", "brand": brand, "label": 1})

    logger.info(f"Generated {len(test_set)} adversarial URLs across {len(BRANDS)} brands")
    return test_set


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    test_set = generate_adversarial_testset()
    df = pd.DataFrame(test_set)

    out_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "adversarial_testset.csv", index=False)

    print(f"\n📊 Summary:")
    print(df["attack_type"].value_counts().to_string())
    print(f"\nTotal: {len(df)} adversarial URLs")
