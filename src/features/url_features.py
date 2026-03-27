"""
PhishGuard++ — URL Feature Extraction
Extracts 20 lexical features from raw URLs for the Tier 1 LightGBM model.

Features:
  1.  url_length            — Total URL character count
  2.  domain_length         — Domain string length
  3.  n_subdomains          — Number of '.' in domain minus 1
  4.  digit_ratio           — Fraction of digits in URL
  5.  special_char_count    — Count of @, !, #, $, %, ^, &, *, etc.
  6.  entropy_domain        — Shannon entropy of domain string
  7.  has_ip_as_domain      — 1 if domain is an IP address
  8.  suspicious_keyword_count — Count of phishing keywords in URL
  9.  tld_in_subdomain      — 1 if a known TLD appears in a subdomain
  10. brand_in_path         — 1 if a popular brand name appears in path
  11. punycode_detected     — 1 if URL contains punycode (xn--)
  12. redirect_count        — Count of '//' in URL (beyond protocol)
  13. https_present         — 1 if scheme is HTTPS
  14. url_depth             — Number of '/' in path
  15. longest_path_token    — Length of the longest path segment
  16. avg_path_token_len    — Average length of path segments
  17. query_digit_count     — Number of digits in query string
  18. dot_count             — Total dots in URL
  19. slash_count           — Total slashes in URL
  20. ampersand_count       — Total '&' characters (query params)
"""

import re
import math
import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs
from typing import Dict, List
import tldextract
import logging

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "account", "update", "secure",
    "banking", "confirm", "password", "credential", "suspend",
    "alert", "urgent", "unlock", "validate", "authenticate",
    "paypal", "apple", "microsoft", "amazon", "google",
    "facebook", "netflix", "instagram", "whatsapp",
]

POPULAR_BRANDS = [
    "paypal", "apple", "microsoft", "amazon", "google", "facebook",
    "netflix", "instagram", "twitter", "linkedin", "dropbox", "adobe",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "dhl", "fedex", "ups", "usps",
    "sbi", "icici", "hdfc", "irctc",  # Indian brands
]

KNOWN_TLDS = [
    "com", "net", "org", "edu", "gov", "info", "biz", "co",
    "io", "me", "in", "uk", "us", "ca", "au", "de", "fr",
]


def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def is_ip_address(domain: str) -> bool:
    """Check if domain is an IP address."""
    # IPv4
    ipv4_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    if re.match(ipv4_pattern, domain):
        return True
    # IPv6 (simplified)
    if ":" in domain and all(c in "0123456789abcdefABCDEF:" for c in domain):
        return True
    return False


def extract_url_features(url: str) -> Dict[str, float]:
    """Extract 20 lexical features from a single URL."""
    features = {}

    # Safely parse URL
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
    except Exception:
        parsed = urlparse(f"https://{url}")

    try:
        extracted = tldextract.extract(url)
    except Exception:
        extracted = tldextract.extract(f"https://{url}")

    domain = extracted.domain + "." + extracted.suffix if extracted.suffix else extracted.domain
    full_domain = f"{extracted.subdomain}.{domain}" if extracted.subdomain else domain
    path = parsed.path or ""
    query = parsed.query or ""

    # 1. url_length
    features["url_length"] = len(url)

    # 2. domain_length
    features["domain_length"] = len(full_domain)

    # 3. n_subdomains
    subdomain_parts = [s for s in extracted.subdomain.split(".") if s]
    features["n_subdomains"] = len(subdomain_parts)

    # 4. digit_ratio
    digits = sum(1 for c in url if c.isdigit())
    features["digit_ratio"] = digits / max(len(url), 1)

    # 5. special_char_count
    special_chars = set("@!#$%^&*()_+=~`|\\{}[]<>?")
    features["special_char_count"] = sum(1 for c in url if c in special_chars)

    # 6. entropy_domain
    features["entropy_domain"] = shannon_entropy(full_domain)

    # 7. has_ip_as_domain
    features["has_ip_as_domain"] = int(is_ip_address(full_domain))

    # 8. suspicious_keyword_count
    url_lower = url.lower()
    features["suspicious_keyword_count"] = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    # 9. tld_in_subdomain
    features["tld_in_subdomain"] = int(
        any(f".{tld}." in f".{extracted.subdomain}." for tld in KNOWN_TLDS)
    ) if extracted.subdomain else 0

    # 10. brand_in_path
    path_lower = path.lower()
    features["brand_in_path"] = int(any(brand in path_lower for brand in POPULAR_BRANDS))

    # 11. punycode_detected
    features["punycode_detected"] = int("xn--" in url.lower())

    # 12. redirect_count — number of '//' beyond protocol
    features["redirect_count"] = max(0, url.count("//") - 1)

    # 13. https_present
    features["https_present"] = int(parsed.scheme == "https")

    # 14. url_depth
    path_segments = [s for s in path.split("/") if s]
    features["url_depth"] = len(path_segments)

    # 15. longest_path_token
    features["longest_path_token"] = max((len(s) for s in path_segments), default=0)

    # 16. avg_path_token_len
    features["avg_path_token_len"] = (
        sum(len(s) for s in path_segments) / len(path_segments)
        if path_segments else 0.0
    )

    # 17. query_digit_count
    features["query_digit_count"] = sum(1 for c in query if c.isdigit())

    # 18. dot_count
    features["dot_count"] = url.count(".")

    # 19. slash_count
    features["slash_count"] = url.count("/")

    # 20. ampersand_count
    features["ampersand_count"] = url.count("&")

    return features


# ── Column order for consistency ───────────────────────────────
URL_FEATURE_NAMES = [
    "url_length", "domain_length", "n_subdomains", "digit_ratio",
    "special_char_count", "entropy_domain", "has_ip_as_domain",
    "suspicious_keyword_count", "tld_in_subdomain", "brand_in_path",
    "punycode_detected", "redirect_count", "https_present", "url_depth",
    "longest_path_token", "avg_path_token_len", "query_digit_count",
    "dot_count", "slash_count", "ampersand_count",
]

HUMAN_READABLE_NAMES = {
    "url_length": "URL is unusually long",
    "domain_length": "Domain name is very long",
    "n_subdomains": "URL has many subdomains",
    "digit_ratio": "URL contains many numbers",
    "special_char_count": "URL has unusual special characters",
    "entropy_domain": "Domain name looks randomly generated",
    "has_ip_as_domain": "Domain is an IP address instead of a name",
    "suspicious_keyword_count": "URL contains suspicious keywords (login, verify, etc.)",
    "tld_in_subdomain": "A domain extension appears in a subdomain",
    "brand_in_path": "A well-known brand name appears in the URL path",
    "punycode_detected": "URL uses punycode encoding (homoglyph attack)",
    "redirect_count": "URL has multiple redirects",
    "https_present": "Site uses HTTPS encryption",
    "url_depth": "URL has an unusually deep path",
    "longest_path_token": "A URL path segment is very long",
    "avg_path_token_len": "URL path segments are long on average",
    "query_digit_count": "Query parameters have many digits",
    "dot_count": "URL has many dots",
    "slash_count": "URL has many slashes",
    "ampersand_count": "URL has many query parameters",
}


def extract_features_batch(urls: pd.Series, show_progress: bool = True) -> pd.DataFrame:
    """Extract URL features for a batch of URLs.

    Args:
        urls: Series of URL strings
        show_progress: Whether to show tqdm progress bar

    Returns:
        DataFrame with 20 feature columns
    """
    from tqdm import tqdm

    iterator = tqdm(urls, desc="Extracting URL features") if show_progress else urls
    records = []
    for url in iterator:
        try:
            features = extract_url_features(str(url))
        except Exception as e:
            logger.warning(f"Failed to extract features from '{url}': {e}")
            features = {name: 0.0 for name in URL_FEATURE_NAMES}
        records.append(features)

    return pd.DataFrame(records, columns=URL_FEATURE_NAMES)


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login.php",
        "https://paypa1-secure-login.suspicious-site.com/verify/account?id=12345",
        "https://xn--pypal-4ve.com/signin",
        "http://www.amazon.com.evil.com/password-reset",
    ]

    print("── URL Feature Extraction Test ──\n")
    for url in test_urls:
        features = extract_url_features(url)
        print(f"URL: {url}")
        for name, val in features.items():
            if val != 0:
                print(f"  {name}: {val}")
        print()
