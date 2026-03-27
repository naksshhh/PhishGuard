# PhishGuard++ — Walkthrough (Phase 1)

## What Was Built

Phase 1 (Foundation & Data Engineering) core files — **11 new files** across 5 directories.

### Project Skeleton

| File | Purpose |
|---|---|
| [requirements.txt](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/requirements.txt) | 25+ packages: ML, DL, FastAPI, Firebase, Gemini |
| [.env.example](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/.env.example) | API key template |
| [.gitignore](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/.gitignore) | Ignores models, creds, caches |
| [README.md](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/README.md) | Project overview + quick start |

---

### Data Pipeline

#### [dataset_builder.py](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/src/data/dataset_builder.py)
- Loads all 5 datasets (Kaggle UCI, PhiUSIIL, PhishTank, Tranco, Mendeley)
- Parses Mendeley [index.sql](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/datasets/Mendeley%20phishing%20dataset/index.sql) dump to extract `url → label → html_filename` mappings
- Deduplicates by URL, normalizes labels (1=phishing, 0=legit)
- Creates stratified 80/10/10 train/val/test splits

---

### Feature Extraction (40 Features Total)

#### [url_features.py](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/src/features/url_features.py) — 20 URL Lexical Features
- Shannon entropy, punycode detection, IP-as-domain check
- Suspicious keyword matching (20 keywords), brand detection (24 brands)
- TLD-in-subdomain detection, redirect counting
- Human-readable names for SHAP XAI explanations

#### [html_features.py](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/src/features/html_features.py) — 20 HTML Structural Features
- External form action, hidden inputs, password fields
- JavaScript anchors, meta refresh, favicon analysis
- External link ratio, social media links, ad network detection
- Smart HTML excerpt extractor for CodeBERT/Gemini input

#### [extract_all.py](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/src/features/extract_all.py) — Orchestrator
- Runs URL + HTML feature extraction on all splits
- Joins Mendeley HTML features to unified dataset
- Outputs `train_features.csv`, `val_features.csv`, `test_features.csv`

---

### Baseline Model Training

#### [baseline_race.py](file:///c:/Users/naksh/OneDrive/Desktop/Sem%206/Capstone/solutions_challenge/src/models/baseline_race.py)
- **Race**: Random Forest vs XGBoost vs LightGBM
- **Optuna**: 50-trial Bayesian hyperparameter sweep on LightGBM
- Logs to both W&B and CSV
- Saves best model + feature importance plot

---

## How to Run Phase 1

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Build unified dataset from all 5 sources
python -m src.data.dataset_builder

# Step 3: Extract 40 features (URL + HTML)
python -m src.features.extract_all

# Step 4: Run baseline race + Optuna optimization
python -m src.models.baseline_race
```

## What's Next (Phase 2)

1. Export winning LightGBM to ONNX (INT8 quantized, <300KB)
2. Build Chrome MV3 extension with on-device ONNX inference
3. Create FastAPI backend with 3 async endpoints
4. Fine-tune DistilBERT ("PhishBERT") and CodeBERT
5. Build attention-weighted meta-classifier fusion
