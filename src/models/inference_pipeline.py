import logging
import asyncio
import base64
import io
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from PIL import Image

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    pass

from src.models.attention_fusion import AttentionFusion
from src.models.efficientnet_visual import PhishEfficientNet

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

class FusionOrchestrator:
    """
    Dynamically loads available SOTA models and aggregates predictions using the AttentionFusion layer.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.phishbert = None
        self.pb_tokenizer = None
        self.codebert = None
        self.cb_tokenizer = None
        self.efficientnet = None
        self.tabular = None
        self.fusion = None
        
        self.load_models()

    def load_models(self):
        logger.info(f"Loading SOTA models to {self.device}...")

        # 0. Tabular (LightGBM Stage 1)
        lgbm_path = MODELS_DIR / "lightgbm_stage1.pkl"
        if lgbm_path.exists():
            try:
                import joblib
                self.tabular = joblib.load(lgbm_path)
                logger.info("✅ Loaded LightGBM Stage 1")
            except Exception as e:
                logger.error(f"Failed to load LightGBM: {e}")
        
        # 1. PhishBERT (Semantic URL)
        pb_path = MODELS_DIR / "phishbert"
        if pb_path.exists():
            try:
                self.pb_tokenizer = AutoTokenizer.from_pretrained(pb_path)
                self.phishbert = AutoModelForSequenceClassification.from_pretrained(pb_path).to(self.device).eval()
                logger.info("✅ Loaded PhishBERT")
            except Exception as e:
                logger.error(f"Failed to load PhishBERT: {e}")

        # 2. CodeBERT (Semantic HTML)
        cb_path = MODELS_DIR / "codebert"
        if cb_path.exists():
            try:
                self.cb_tokenizer = AutoTokenizer.from_pretrained(cb_path)
                self.codebert = AutoModelForSequenceClassification.from_pretrained(cb_path).to(self.device).eval()
                logger.info("✅ Loaded CodeBERT")
            except Exception as e:
                logger.error(f"Failed to load CodeBERT: {e}")

        # 3. EfficientNet (Visual)
        enet_path = MODELS_DIR / "efficientnet_v1.pth"
        if enet_path.exists():
            try:
                model = PhishEfficientNet("efficientnet_b7")
                model.load_state_dict(torch.load(enet_path, map_location=self.device, weights_only=True))
                self.efficientnet = model.to(self.device).eval()
                logger.info("✅ Loaded EfficientNet-B7")
            except Exception as e:
                logger.error(f"Failed to load EfficientNet: {e}")

        # 4. Attention Fusion Core
        fusion_path = MODELS_DIR / "attention_fusion.pth"
        if fusion_path.exists():
            try:
                model = AttentionFusion(n_branches=4)
                model.load_state_dict(torch.load(fusion_path, map_location=self.device, weights_only=True))
                self.fusion = model.to(self.device).eval()
                logger.info("✅ Loaded Attention-Fusion Core")
            except Exception as e:
                logger.error(f"Failed to load Fusion layer: {e}")

    async def get_tabular_score(self, features: dict):
        """Tier 2 Cloud (LightGBM)"""
        if not self.tabular:
            return 0.5
            
        try:
            from src.explainability.shap_pipeline import FEATURE_ORDER
            def _predict():
                df = pd.DataFrame([features], columns=FEATURE_ORDER)
                if hasattr(self.tabular, "predict_proba"):
                    return self.tabular.predict_proba(df)[0][1]
                return self.tabular.predict(df)[0]
                
            score = await asyncio.to_thread(_predict)
            return float(score)
        except Exception as e:
            logger.error(f"Tabular prediction failed: {e}")
            return 0.5

    async def get_phishbert_score(self, url: str):
        if not self.phishbert:
            return 0.5, 0.0 # score, mask
        def _predict():
            inputs = self.pb_tokenizer(url, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.phishbert(**inputs).logits
                prob = F.softmax(logits, dim=-1)[0][1].item()
            return prob
        return await asyncio.to_thread(_predict), 1.0

    async def get_codebert_score(self, html_excerpt: str):
        if not self.codebert or not html_excerpt or len(html_excerpt.strip()) < 100:
            return 0.5, 0.0 # Missing or meaningless HTML
        def _predict():
            inputs = self.cb_tokenizer(html_excerpt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.codebert(**inputs).logits
                prob = F.softmax(logits, dim=-1)[0][1].item()
            return prob
        return await asyncio.to_thread(_predict), 1.0

    def _predict_visual_b64(self, b64_screenshot: str):
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # Decode B64
            if "," in b64_screenshot:
                b64_screenshot = b64_screenshot.split(",")[1]
            img_data = base64.b64decode(b64_screenshot)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda') if self.device.type == 'cuda' else torch.no_grad():
                    logits = self.efficientnet(tensor)
                    prob = torch.sigmoid(logits).item()
            return prob
        except Exception as e:
            logger.error(f"Visual b64 parse error: {e}")
            return 0.5

    async def get_efficientnet_score(self, b64_screenshot: str):
        if not self.efficientnet or not b64_screenshot:
            return 0.5, 0.0
        score = await asyncio.to_thread(self._predict_visual_b64, b64_screenshot)
        return score, 1.0

    def _predict_visual_path(self, image_path: Path):
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda') if self.device.type == 'cuda' else torch.no_grad():
                    logits = self.efficientnet(tensor)
                    prob = torch.sigmoid(logits).item()
            return prob
        except Exception as e:
            logger.error(f"Visual path parse error: {e}")
            return 0.5

    async def get_efficientnet_score_from_path(self, image_path: Path):
        if not self.efficientnet or not image_path.exists():
            return 0.5, 0.0
        score = await asyncio.to_thread(self._predict_visual_path, image_path)
        return score, 1.0

    async def fuse_predictions(self, tabular_score: float, url: str, html: str, screenshot: str):
        """
        Runs all local branches predicting Phase 1/2 probability, then fusing them.
        """
        # Run local heavy models concurrently
        pb_task = self.get_phishbert_score(url)
        cb_task = self.get_codebert_score(html)
        en_task = self.get_efficientnet_score(screenshot)
        
        (pb_s, pb_m), (cb_s, cb_m), (en_s, en_m) = await asyncio.gather(pb_task, cb_task, en_task)
        
        # 1. Tabular (Always active)
        # 2. PhishBERT (Always active if model loaded)
        # 3. CodeBERT (Maskable)
        # 4. EfficientNet (Maskable)
        
        scores = torch.tensor([[tabular_score, pb_s, cb_s, en_s]], dtype=torch.float32).to(self.device)
        masks = torch.tensor([[1.0, pb_m, cb_m, en_m]], dtype=torch.float32).to(self.device)

        if self.fusion:
            with torch.no_grad():
                fused_prob = self.fusion(scores, masks).item()
        else:
            # Fallback heuristic if fusion model not trained yet:
            active_scores = [s for s, m in zip([tabular_score, pb_s, cb_s, en_s], [1.0, pb_m, cb_m, en_m]) if m > 0.5]
            fused_prob = sum(active_scores) / len(active_scores) if active_scores else 0.5

        return {
            "fused_score": fused_prob,
            "branches": {
                "tabular": tabular_score,
                "phishbert": pb_s if pb_m else None,
                "codebert": cb_s if cb_m else None,
                "efficientnet": en_s if en_m else None
            }
        }

# Global singleton so models are loaded only once on startup
orchestrator = FusionOrchestrator()
