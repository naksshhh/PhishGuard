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

    def forward(self, branch_scores: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            branch_scores: (batch, n_branches) — probabilities from each branch
            mask: (batch, n_branches) — 1.0 for active branch, 0.0 for missing
        Returns:
            fused_score: (batch, 1) — final phishing probability
        """
        # 1. Calculate Attention Weights
        attn_logits = self.attention[:3](branch_scores) # Linear + ReLU + Linear
        
        # 2. Apply Mask if provided
        if mask is not None:
            # Shift masked logits to a very large negative value so softmax is 0
            attn_logits = attn_logits + (1.0 - mask) * -1e9
            
        weights = torch.softmax(attn_logits, dim=-1)
        
        # 3. Apply weights and classify
        weighted = branch_scores * weights
        return self.classifier(weighted)

    def get_attention_weights(self, branch_scores: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Expose attention weights for interpretability."""
        attn_logits = self.attention[:3](branch_scores)
        if mask is not None:
            attn_logits = attn_logits + (1.0 - mask) * -1e9
        return torch.softmax(attn_logits, dim=-1)


def train_fusion(
    data_path: Path,
    epochs: int = 500,
    batch_size: int = 256,
    lr: float = 2e-3,
):
    """
    Train the attention fusion with high-performance optimization strategies.
    """
    import pandas as pd
    from torch.utils.data import TensorDataset, DataLoader
    
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples for SOTA fusion training.")

    # Prepare Tensors
    scores = df[["tabular_score", "url_score", "html_score", "visual_score"]].values
    masks = df[["tabular_score", "url_score", "html_mask", "visual_mask"]].copy()
    masks["tabular_score"] = 1.0
    masks["url_score"] = 1.0
    masks = masks.values
    labels = df["label"].values

    X_scores = torch.tensor(scores, dtype=torch.float32)
    X_masks = torch.tensor(masks, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Train/Val Split (80/20 for better verification)
    total_samples = len(df)
    indices = torch.randperm(total_samples)
    split = int(0.8 * total_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(X_scores[train_idx], X_masks[train_idx], y[train_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model & Optimization
    n_branches = scores.shape[1]
    model = AttentionFusion(n_branches=n_branches)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    best_state = None
    patience = 40
    no_improve = 0

    logger.info(f"Starting optimized training (500 epochs, batch_size={batch_size})...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for b_scores, b_masks, b_labels in train_loader:
            optimizer.zero_grad()
            pred = model(b_scores, b_masks)
            loss = criterion(pred, b_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()

        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_scores[val_idx], X_masks[val_idx])
                val_acc = ((val_pred > 0.5).float() == y[val_idx]).float().mean().item()
                val_loss = criterion(val_pred, y[val_idx]).item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = model.state_dict().copy()
                    no_improve = 0
                else:
                    no_improve += 10 # Since we check every 10

            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4%}")

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4%}")
            break

    # Save Best Model
    if best_state:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, MODELS_DIR / "attention_fusion.pth")
        logger.info(f"✅ SOTA Fusion Layer saved with Best Val Acc: {best_val_acc:.4%}")
    
    return model

if __name__ == "__main__":
    DATA_PATH = BASE_DIR / "datasets" / "processed" / "fusion_train_v2.csv"
    train_fusion(DATA_PATH)
