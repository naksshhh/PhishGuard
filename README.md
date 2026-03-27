# PhishGuard++

> Multi-modal phishing detection Chrome extension with 3-tier cascade architecture.  
> Google Solutions Challenge 2026 · Capstone Project

## 🏗 Architecture

```
URL Input → Community Cache (Firebase) → Tier 1 (ONNX LightGBM, <15ms)
         → Tier 2 (FastAPI: XGBoost + DistilBERT + CodeBERT, <200ms)
         → Tier 3 (Gemini 1.5 Flash multimodal, <1500ms)
         → Risk Score + Plain-English Explanation
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment config
cp .env.example .env
# Edit .env with your API keys

# 3. Build dataset
python -m src.data.dataset_builder

# 4. Extract features (40 URL + HTML features)
python -m src.features.extract_all

# 5. Train baseline models (RF vs XGB vs LightGBM + Optuna)
python -m src.models.baseline_race
```

## 📂 Project Structure

```
src/
├── data/           → Dataset building & augmentation
├── features/       → Feature extraction (URL + HTML)
├── models/         → ML models & training
├── explainability/ → SHAP + XAI pipeline
└── evaluation/     → Ablation study & benchmarks
backend/            → FastAPI server (Tier 2 + 3)
extension/          → Chrome MV3 extension (Tier 1)
datasets/           → Raw & processed data
models/             → Trained model artifacts
```

## 📊 Datasets

| Source | Samples | Type |
|--------|---------|------|
| PhiUSIIL | ~135k | URLs + features |
| Mendeley | ~80k | URLs + HTML files |
| PhishTank | ~30k | Verified phishing |
| Tranco | 1M | Legitimate domains |
| Kaggle UCI | ~11k | Pre-computed features |

## 🎯 Novel Contributions

1. **Hierarchical Latency-Aware Cascade** — 95% URLs resolved before any server call
2. **True In-Browser ONNX Inference** — INT8-quantized LightGBM via WebAssembly
3. **Gemini Brand-Impersonation Reasoning** — Screenshot + HTML → brand verification
4. **Community Intelligence Layer** — Firebase crowd-sourced zero-day detection
