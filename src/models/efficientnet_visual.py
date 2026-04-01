import logging
from contextlib import nullcontext
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "screenshots"
METADATA_PATH = BASE_DIR / "datasets" / "processed" / "mendeley_metadata.csv"
MODELS_DIR = BASE_DIR / "models"

class ScreenshotDataset(Dataset):
    def __init__(self, data_dir: Path, meta_path: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load labels and map to screenshots
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}")
            
        df = pd.read_csv(meta_path)
        # Assuming meta file has url, label, and we can link to screenshot via filename
        self.samples = []
        for _, row in df.iterrows():
            # Link filename/stem to image. Mendeley files are .html, screenshots are .jpg
            img_name = row["html_filename"].replace(".html", ".jpg")
            img_path = data_dir / img_name
            if img_path.exists():
                self.samples.append((img_path, int(row["label"])))
        
        logger.info(f"Loaded {len(self.samples)} valid screenshots.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class PhishEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet_b7"):
        super(PhishEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet-B7
        if model_name == "efficientnet_b7":
            self.backbone = models.efficientnet_b7(weights='IMAGENET1K_V1')
            # Extract number of input features for the classifier
            num_ftrs = self.backbone.classifier[1].in_features
            # Replace classifier with a binary head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(num_ftrs, 1)
            )
        else:
            # Fallback to B0 if B7 is too memory-heavy for current VRAM
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(num_ftrs, 1)
            )

    def forward(self, x):
        return self.backbone(x)

def train_visual(epochs: int = 10, batch_size: int = 4):
    """Fine-tune EfficientNet on website screenshots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    
    # Image Augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = ScreenshotDataset(DATA_DIR, METADATA_PATH, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Use B0 if VRAM is tight, otherwise B7
    model = PhishEfficientNet(model_name="efficientnet_b7").to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed-Precision for 4GB VRAM optimization
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    logger.info("Starting Visual ML Training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Using torch lib instead of deprecated torch.cuda.amp
            with torch.amp.autocast('cuda') if scaler else nullcontext():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "efficientnet_v1.pth")
    logger.info("Saved Visual ML model to models/efficientnet_v1.pth")

if __name__ == "__main__":
    train_visual()
