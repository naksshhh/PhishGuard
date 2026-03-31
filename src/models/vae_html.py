"""
PhishGuard++ — VAE for HTML Latent Features
A Variational Autoencoder to compress high-dimensional HTML structural 
features into a dense 32-dim latent representation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "models"


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Features should be normalized to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_vae(epochs: int = 50, batch_size: int = 64, latent_dim: int = 32):
    logger.info(f"Training VAE (latent_dim={latent_dim})...")
    
    # Load HTML features
    # (In Phase 1, we use the 20 pre-extracted HTML features)
    html_features_path = DATA_DIR / "mendeley_html_features.csv"
    if not html_features_path.exists():
        logger.error(f"HTML features not found. Run feature extraction first.")
        return

    df = pd.read_csv(html_features_path)
    # Filter out non-feature cols
    data = df.drop(columns=["html_filename"], errors="ignore").values.astype(np.float32)
    
    # Normalize data to [0, 1] for Sigmoid decoder
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    input_dim = data_scaled.shape[1]
    tensor_data = torch.from_numpy(data_scaled)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = VAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(dataloader.dataset):.6f}")

    # Save model and scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "vae_html.pth")
    import joblib
    joblib.dump(scaler, MODELS_DIR / "vae_scaler.pkl")
    
    logger.info(f"Saved VAE model to {MODELS_DIR / 'vae_html.pth'}")
    return model, scaler


if __name__ == "__main__":
    train_vae()
