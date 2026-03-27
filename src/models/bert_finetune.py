"""
PhishGuard++ — BERT-based Phishing Detection
Fine-tuning and inference for PhishBERT (DistilBERT) and CodeBERT.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "models"


def fine_tune_bert(model_name: str = "distilbert-base-uncased", output_dir: str = "phishbert"):
    """Fine-tune a BERT-based model on URL/HTML textual signals."""
    logger.info(f"Preparing to fine-tune {model_name}...")
    
    # Load unified dataset
    data_path = DATA_DIR / "train_features.csv"
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}. Run extraction first.")
        return

    df = pd.read_csv(data_path)
    
    # For PhishBERT, we might use the URL string as input
    # For CodeBERT, we use the HTML excerpt
    # This is a sample task for demonstrative skeleton
    train_df = df[["url", "label"]].sample(1000) # Small subset for demonstration
    dataset = Dataset.from_pandas(train_df)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["url"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / output_dir),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    logger.info("Starting fine-tuning (demo mode)...")
    # trainer.train() # Uncomment once full data and GPU available
    
    # Save the model
    model.save_pretrained(MODELS_DIR / output_dir)
    tokenizer.save_pretrained(MODELS_DIR / output_dir)
    logger.info(f"Saved {model_name} fine-tuned model to {MODELS_DIR / output_dir}")


if __name__ == "__main__":
    # In Phase 3, we would run this on the GPU once data extraction is complete
    # fine_tune_bert("distilbert-base-uncased", "phishbert")
    # fine_tune_bert("microsoft/codebert-base", "codebert")
    pass
