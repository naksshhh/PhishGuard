"""
PhishGuard++ — Email Dataset Builder

Merges four public email datasets into a single balanced corpus for
RoBERTa fine-tuning:

1. Enron Email Dataset (~500k emails, legitimate class) — Kaggle
2. Apache SpamAssassin Public Corpus (~6k emails, spam/phishing class)
3. Nazario Phishing Corpus 2015 (~2.4k, phishing class)
4. CEAS_08 Dataset (~40k mixed, labeled)

Output: datasets/processed/email_corpus.csv
Columns: subject | sender | body | label (0=legit, 1=phish) | source

Target: ~50k balanced samples (25k legit, 25k phish)
"""

import csv
import email
import email.policy
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "processed"


def _clean_text(text: Optional[str], max_len: int = 5000) -> str:
    """Strip HTML, normalize whitespace, and truncate."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _parse_eml(file_path: Path) -> dict:
    """Parse a single .eml or raw email file into structured fields."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f, policy=email.policy.default)

        subject = str(msg.get("Subject", ""))
        sender = str(msg.get("From", ""))

        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
            if not body:
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="ignore")
                            break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")

        return {
            "subject": _clean_text(subject, 500),
            "sender": sender.strip()[:200],
            "body": _clean_text(body),
        }
    except Exception as e:
        logger.debug(f"Failed to parse {file_path}: {e}")
        return None


# ── Dataset-Specific Loaders ─────────────────────────────────


def load_enron(max_samples: int = 25000) -> pd.DataFrame:
    """
    Load Enron email dataset (legitimate emails).
    Expected location: datasets/enron/maildir/
    Download: https://www.cs.cmu.edu/~enron/
    """
    enron_dir = DATASETS_DIR / "enron" / "maildir"
    if not enron_dir.exists():
        logger.warning(
            f"Enron dataset not found at {enron_dir}. "
            f"Download from https://www.cs.cmu.edu/~enron/"
        )
        return pd.DataFrame()

    records = []
    count = 0

    for eml_path in enron_dir.rglob("*"):
        if count >= max_samples:
            break
        if eml_path.is_file() and eml_path.stat().st_size < 100_000:
            parsed = _parse_eml(eml_path)
            if parsed and len(parsed["body"]) > 50:
                parsed["label"] = 0
                parsed["source"] = "enron"
                records.append(parsed)
                count += 1

    logger.info(f"Enron: Loaded {len(records)} legitimate emails")
    return pd.DataFrame(records)


def load_spamassassin() -> pd.DataFrame:
    """
    Load Apache SpamAssassin public corpus.
    Expected location: datasets/spamassassin/
    Subdirs: easy_ham/, hard_ham/, spam/
    Download: https://spamassassin.apache.org/old/publiccorpus/
    """
    sa_dir = DATASETS_DIR / "spamassassin"
    if not sa_dir.exists():
        logger.warning(
            f"SpamAssassin dataset not found at {sa_dir}. "
            f"Download from https://spamassassin.apache.org/old/publiccorpus/"
        )
        return pd.DataFrame()

    records = []

    # Legitimate emails (ham)
    for ham_dir in ["easy_ham", "easy_ham_2", "hard_ham"]:
        ham_path = sa_dir / ham_dir
        if ham_path.exists():
            for f in ham_path.iterdir():
                if f.is_file() and f.name != "cmds":
                    parsed = _parse_eml(f)
                    if parsed and len(parsed["body"]) > 30:
                        parsed["label"] = 0
                        parsed["source"] = "spamassassin"
                        records.append(parsed)

    # Spam/phishing emails
    for spam_dir in ["spam", "spam_2"]:
        spam_path = sa_dir / spam_dir
        if spam_path.exists():
            for f in spam_path.iterdir():
                if f.is_file() and f.name != "cmds":
                    parsed = _parse_eml(f)
                    if parsed and len(parsed["body"]) > 30:
                        parsed["label"] = 1
                        parsed["source"] = "spamassassin"
                        records.append(parsed)

    logger.info(f"SpamAssassin: Loaded {len(records)} emails")
    return pd.DataFrame(records)


def load_nazario() -> pd.DataFrame:
    """
    Load Nazario Phishing Corpus (phishing emails).
    Expected location: datasets/nazario/
    Contains .eml files of verified phishing emails.
    """
    nazario_dir = DATASETS_DIR / "nazario"
    if not nazario_dir.exists():
        logger.warning(f"Nazario dataset not found at {nazario_dir}")
        return pd.DataFrame()

    records = []
    for eml_path in nazario_dir.rglob("*.eml"):
        parsed = _parse_eml(eml_path)
        if parsed and len(parsed["body"]) > 30:
            parsed["label"] = 1
            parsed["source"] = "nazario"
            records.append(parsed)

    for eml_path in nazario_dir.rglob("*.txt"):
        parsed = _parse_eml(eml_path)
        if parsed and len(parsed["body"]) > 30:
            parsed["label"] = 1
            parsed["source"] = "nazario"
            records.append(parsed)

    logger.info(f"Nazario: Loaded {len(records)} phishing emails")
    return pd.DataFrame(records)


def load_ceas08() -> pd.DataFrame:
    """
    Load CEAS_08 dataset (mixed labeled emails).
    Expected location: datasets/ceas08/ceas08.csv
    Columns expected: sender, subject, body, label
    """
    ceas_path = DATASETS_DIR / "ceas08" / "ceas08.csv"
    if not ceas_path.exists():
        logger.warning(f"CEAS_08 dataset not found at {ceas_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(ceas_path, encoding="utf-8", on_bad_lines="skip")

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Map labels
        if "label" in df.columns:
            df["label"] = df["label"].apply(
                lambda x: 1 if str(x).lower() in ("spam", "phishing", "1") else 0
            )

        # Standardize columns
        for col in ["subject", "sender", "body"]:
            if col not in df.columns:
                df[col] = ""

        df["body"] = df["body"].apply(lambda x: _clean_text(str(x)))
        df["subject"] = df["subject"].apply(lambda x: _clean_text(str(x), 500))
        df["source"] = "ceas08"

        df = df[df["body"].str.len() > 30]

        logger.info(f"CEAS_08: Loaded {len(df)} emails")
        return df[["subject", "sender", "body", "label", "source"]]

    except Exception as e:
        logger.error(f"Failed to load CEAS_08: {e}")
        return pd.DataFrame()


# ── Main Pipeline ─────────────────────────────────────────────


def build_email_corpus(
    target_size: int = 50000,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge all datasets, balance classes, deduplicate, and save.

    Args:
        target_size: Target total samples (balanced 50/50)
        output_path: Where to save the CSV

    Returns:
        The final balanced DataFrame
    """
    output_path = output_path or OUTPUT_DIR / "email_corpus.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building unified email corpus...")
    logger.info("=" * 60)

    # Load all datasets
    dfs = []
    for loader in [load_enron, load_spamassassin, load_nazario, load_ceas08]:
        df = loader()
        if not df.empty:
            dfs.append(df)

    if not dfs:
        logger.error("No datasets found! Please download at least one dataset.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined raw: {len(combined)} emails")

    # Deduplicate on body text (fuzzy: first 200 chars)
    combined["body_key"] = combined["body"].str[:200]
    combined = combined.drop_duplicates(subset=["body_key"])
    combined = combined.drop(columns=["body_key"])
    logger.info(f"After dedup: {len(combined)} emails")

    # Class distribution
    class_counts = combined["label"].value_counts()
    logger.info(f"Class distribution:\n{class_counts}")

    # Balance classes
    half = target_size // 2
    legit = combined[combined["label"] == 0]
    phish = combined[combined["label"] == 1]

    n_legit = min(len(legit), half)
    n_phish = min(len(phish), half)

    balanced = pd.concat(
        [
            legit.sample(n=n_legit, random_state=42),
            phish.sample(n=n_phish, random_state=42),
        ],
        ignore_index=True,
    )

    # Shuffle
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Final balanced corpus: {len(balanced)} emails")
    logger.info(f"  Legit: {(balanced['label'] == 0).sum()}")
    logger.info(f"  Phish: {(balanced['label'] == 1).sum()}")

    # Save
    balanced.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Saved to: {output_path}")

    return balanced


if __name__ == "__main__":
    build_email_corpus()
