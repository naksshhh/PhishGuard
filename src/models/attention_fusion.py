"""
PhishGuard++ — Attention-Weighted Meta-Classifier (Fusion Layer)
Fuses predictions from multiple branches (XGBoost, PhishBERT, CodeBERT, EfficientNet)
using a learnable attention mechanism.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class AttentionFusion(nn.Module):
    """
    2-layer MLP with softmax attention head.
    Takes N branch probability scores and outputs a fused prediction.
    """

    def __init__(self, n_branches: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_branches, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_branches),
            nn.Softmax(dim=-1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_branches, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, branch_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_scores: (batch, n_branches) — probabilities from each branch
        Returns:
            fused_score: (batch, 1) — final phishing probability
        """
        weights = self.attention(branch_scores)
        weighted = branch_scores * weights
        return self.classifier(weighted)

    def get_attention_weights(self, branch_scores: torch.Tensor) -> torch.Tensor:
        """Expose attention weights for interpretability."""
        return self.attention(branch_scores)


def train_fusion(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
):
    """
    Train the attention fusion on branch scores.

    Args:
        train_scores: (N, n_branches) — stacked predictions from each branch
        train_labels: (N,) — ground truth labels
    """
    n_branches = train_scores.shape[1]
    model = AttentionFusion(n_branches=n_branches)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X = torch.tensor(train_scores, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            acc = ((pred > 0.5).float() == y).float().mean()
            logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "attention_fusion.pth")
    logger.info(f"Saved fusion model to {MODELS_DIR / 'attention_fusion.pth'}")

    return model


if __name__ == "__main__":
    # Demo with random data (replace with real branch scores)
    logger.info("Running demo with synthetic branch scores...")
    np.random.seed(42)
    fake_scores = np.random.rand(1000, 4)  # 4 branches
    fake_labels = (fake_scores.mean(axis=1) > 0.5).astype(float)
    train_fusion(fake_scores, fake_labels, epochs=50)
