"""
PhishGuard++ — RoBERTa Email Phishing Classifier (Branch 2)

Fine-tunes roberta-base on a merged email phishing corpus, augmented
with a 6-dimensional Cialdini persuasion feature vector appended to
the [CLS] embedding. Achieves SOTA 99.08% accuracy (MDPI 2024).

Architecture:
    [CLS] embedding (768) ‖ Cialdini features (6) → Linear(774, 256)
    → ReLU → Dropout(0.3) → Linear(256, 2) → Softmax

This module contains:
    - Model definition (RoBERTaEmailClassifier)
    - Training loop (for Colab notebook or local)
    - Inference function (for backend)

Usage:
    Training: Run this file or use the Colab notebook
    Inference: Import RoBERTaEmailInference from backend
"""

import logging
import os
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


# ══════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════


class RoBERTaEmailClassifier(nn.Module):
    """
    RoBERTa-base with a custom classification head that fuses the
    [CLS] token embedding with Cialdini persuasion features.
    """

    def __init__(
        self,
        roberta_model_name: str = "roberta-base",
        n_cialdini: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_labels: int = 2,
    ):
        super().__init__()
        from transformers import RobertaModel

        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        roberta_dim = self.roberta.config.hidden_size  # 768

        # Custom head: [CLS](768) + Cialdini(6) → 774 → 256 → 2
        self.classifier = nn.Sequential(
            nn.Linear(roberta_dim + n_cialdini, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cialdini_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) tokenized email text
            attention_mask: (batch, seq_len)
            cialdini_features: (batch, 6) persuasion principle scores

        Returns:
            logits: (batch, 2)
        """
        outputs = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Concatenate [CLS] + Cialdini features
        combined = torch.cat([cls_embedding, cialdini_features], dim=-1)

        return self.classifier(combined)


# ══════════════════════════════════════════════════════════════
# Training Pipeline (designed for Colab notebook)
# ══════════════════════════════════════════════════════════════


def train_roberta_email(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 512,
    device: Optional[str] = None,
):
    """
    Fine-tune RoBERTa on the email phishing corpus.

    This function is designed to be called from a Colab notebook
    or run standalone. The trained model is saved to output_dir.
    """
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from transformers import RobertaTokenizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Lazy import to avoid circular dependency
    import sys

    sys.path.insert(0, str(BASE_DIR))
    from src.features.cialdini_features import extract_cialdini_features

    data_path = data_path or BASE_DIR / "datasets" / "processed" / "email_corpus.csv"
    output_dir = output_dir or MODELS_DIR / "roberta_email"
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.info(f"Training RoBERTa Email Classifier on {device}")
    logger.info(f"Data: {data_path}")

    # ── Load Data ────────────────────────────────────────────
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Pre-compute Cialdini features for all samples
    logger.info("Extracting Cialdini persuasion features...")
    df["cialdini"] = df["body"].apply(
        lambda x: extract_cialdini_features(str(x))
    )

    # Combine subject + body for the text input
    df["text"] = df.apply(
        lambda r: f"{r.get('subject', '')} [SEP] {r.get('body', '')}",
        axis=1,
    )

    # Train/val/test split
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.12, random_state=42, stratify=train_df["label"]
    )

    logger.info(
        f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    class EmailDataset(Dataset):
        def __init__(self, dataframe):
            self.texts = dataframe["text"].tolist()
            self.labels = dataframe["label"].tolist()
            self.cialdini = dataframe["cialdini"].tolist()

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "cialdini": torch.tensor(
                    self.cialdini[idx], dtype=torch.float32
                ),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_loader = DataLoader(
        EmailDataset(train_df), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        EmailDataset(val_df), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        EmailDataset(test_df), batch_size=batch_size, shuffle=False
    )

    # ── Model ────────────────────────────────────────────────
    model = RoBERTaEmailClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps
    )

    best_val_f1 = 0.0

    # ── Training Loop ────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cialdini = batch["cialdini"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, cialdini)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cialdini = batch["cialdini"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask, cialdini)
                preds = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        from sklearn.metrics import f1_score

        val_f1 = f1_score(val_labels, val_preds, average="binary")
        val_acc = sum(
            1 for p, l in zip(val_preds, val_labels) if p == l
        ) / len(val_labels)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save model
            torch.save(model.state_dict(), output_dir / "model.pth")
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  ✅ Best model saved (F1={val_f1:.4f})")

    # ── Test Evaluation ──────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Final Test Evaluation")

    # Load best model
    model.load_state_dict(
        torch.load(output_dir / "model.pth", map_location=device)
    )
    model.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cialdini = batch["cialdini"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, cialdini)
            preds = logits.argmax(dim=-1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(labels.cpu().tolist())

    report = classification_report(
        test_labels, test_preds, target_names=["Legitimate", "Phishing"]
    )
    logger.info(f"\n{report}")

    # Save config for inference
    config = {
        "model_name": "roberta-base",
        "n_cialdini": 6,
        "hidden_dim": 256,
        "dropout": 0.3,
        "num_labels": 2,
        "max_length": max_length,
        "best_val_f1": best_val_f1,
    }
    import json

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training complete. Model saved to {output_dir}")
    return model


# ══════════════════════════════════════════════════════════════
# Inference (used by the backend)
# ══════════════════════════════════════════════════════════════


class RoBERTaEmailInference:
    """
    Loads a fine-tuned RoBERTa email classifier for inference.
    Singleton pattern — model loaded once on first call.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = 512

        model_dir = MODELS_DIR / "roberta_email"
        if not model_dir.exists():
            logger.warning(
                f"RoBERTa email model not found at {model_dir}. "
                f"Branch 2 will be unavailable."
            )
            self._initialized = True
            return

        try:
            import json
            from transformers import RobertaTokenizer

            # Load config
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                self.max_length = config.get("max_length", 512)

            # Load model
            self.model = RoBERTaEmailClassifier()
            state_dict = torch.load(
                model_dir / "model.pth",
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()

            # Load tokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)

            self._initialized = True
            logger.info("✅ RoBERTa Email Classifier loaded for inference")

        except Exception as e:
            logger.error(f"Failed to load RoBERTa email model: {e}")
            self.model = None
            self.tokenizer = None
            self._initialized = True

    def predict(
        self,
        subject: str,
        body: str,
        cialdini_features: List[float],
    ) -> tuple[float, str]:
        """
        Predict phishing probability for a single email.

        Args:
            subject: Email subject
            body: Email body text
            cialdini_features: 6-dim persuasion vector

        Returns:
            (score, verdict): score 0-1, verdict PHISH/SAFE
        """
        if not self.model or not self.tokenizer:
            return 0.5, "UNAVAILABLE"

        try:
            text = f"{subject} [SEP] {body}"

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            cialdini_tensor = torch.tensor(
                [cialdini_features], dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, cialdini_tensor)
                probs = F.softmax(logits, dim=-1)
                phish_prob = probs[0][1].item()

            verdict = "PHISH" if phish_prob > 0.5 else "SAFE"
            return phish_prob, verdict

        except Exception as e:
            logger.error(f"RoBERTa email inference failed: {e}")
            return 0.5, "ERROR"


# Global singleton
roberta_inference = RoBERTaEmailInference()


if __name__ == "__main__":
    train_roberta_email()
