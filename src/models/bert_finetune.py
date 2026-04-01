import logging
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
HTML_DIR = BASE_DIR / "datasets" / "Mendeley phishing dataset"
MODELS_DIR = BASE_DIR / "models"

def load_data(model_type):
    # Load unified dataset
    data_path = DATA_DIR / "train_features.csv"
    if not data_path.exists():
        logger.error(f"Training data not found at {data_path}. Run extraction first.")
        return None

    df = pd.read_csv(data_path)
    
    if model_type == "phishbert":
        # Make sure we have label and url
        if "url" not in df.columns or "label" not in df.columns:
            logger.error("train_features.csv missing 'url' or 'label' columns.")
            return None
        # Drop nan URLs
        df = df.dropna(subset=["url", "label"])
        return df[["url", "label"]]
        
    elif model_type == "codebert":
        logger.info("Loading Mendeley metadata to join HTML filenames...")
        meta_path = DATA_DIR / "mendeley_metadata.csv"
        if not meta_path.exists():
            logger.error(f"Metadata not found at {meta_path}")
            return None
            
        meta_df = pd.read_csv(meta_path)
        # Join with metadata to get html_filename using URL
        joined_df = df.merge(meta_df[["url", "html_filename"]], on="url", how="inner")
        joined_df = joined_df.dropna(subset=["html_filename", "label"])
        
        logger.info(f"Joined data size with HTML mapping: {len(joined_df)}")
        
        # Function to read HTML snippet
        def get_html_snippet(filename):
            try:
                filepath = HTML_DIR / filename
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read(2048) # Read just enough to fill 512 tokens
            except Exception:
                pass
            return ""
        
        logger.info("Extracting HTML snippets from files in parallel... (This should be much faster)")
        from concurrent.futures import ThreadPoolExecutor
        
        # Parallel file reading to avoid single-threaded I/O bottleneck
        def wrap_get_snippet(row):
            return get_html_snippet(row["html_filename"])

        with ThreadPoolExecutor(max_workers=16) as executor:
            # Convert to list and run in parallel
            html_contents = list(executor.map(get_html_snippet, joined_df["html_filename"].tolist()))
            
        joined_df["html_content"] = html_contents
        
        # Filter out rows where HTML couldn't be loaded
        joined_df = joined_df[joined_df["html_content"].str.len() > 0]
        logger.info(f"Rows with valid HTML: {len(joined_df)}")
        
        # Rename html_content to text column so tokenizer function is generalized
        joined_df = joined_df.rename(columns={"html_content": "text"})
        return joined_df[["text", "label"]]
        
    return None

def fine_tune_bert(model_type: str = "phishbert"):
    if model_type == "phishbert":
        model_name = "distilbert-base-uncased"
        output_dir = "phishbert"
        text_column = "url"
    else:
        model_name = "microsoft/codebert-base"
        output_dir = "codebert"
        text_column = "text"

    logger.info(f"Preparing to fine-tune {model_name} as {model_type}...")
    
    df = load_data(model_type)
    if df is None or len(df) == 0:
        logger.error("No valid data loaded. Exiting.")
        return
        
    if model_type == "phishbert":
        df = df.rename(columns={"url": "text"})
        
    logger.info(f"Total training samples: {len(df)}")
    
    dataset = Dataset.from_pandas(df)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Model
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 4GB VRAM Optimizer config
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / output_dir),
        eval_strategy="no", # We don't have val set split in this script for brevity, SOTA requires we train on train set 
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Small batch size for 4GB VRAM
        gradient_accumulation_steps=8, # Effective batch size = 32
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=True, # Mixed precision for less VRAM
        dataloader_num_workers=2,
        logging_steps=50,
        report_to="wandb", # Enable Weights & Biases tracking
        run_name=f"{model_type}_finetuning",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    logger.info("Starting fine-tuning...")
    trainer.train() 
    
    # Save the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODELS_DIR / output_dir)
    tokenizer.save_pretrained(MODELS_DIR / output_dir)
    logger.info(f"Saved {model_name} fine-tuned model to {MODELS_DIR / output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["phishbert", "codebert"], required=True, help="Model to fine-tune")
    args = parser.parse_args()
    
    fine_tune_bert(args.model)
