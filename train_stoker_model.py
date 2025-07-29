#!/usr/bin/env python3
"""
Fine-tune a language model on Bram Stoker's writing style using Dracula text.
Uses GPT-2 as base model with Hugging Face Transformers.
"""

import os
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
import json
from tqdm import tqdm

class DraculaDataset:
    def __init__(self, chunks_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read chunks
        print(f"Loading chunks from {chunks_file}...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = [line.strip().replace('\\n', '\n') for line in f if line.strip()]
        
        print(f"Loaded {len(self.chunks)} text chunks")
        
        # Tokenize all chunks
        print("Tokenizing chunks...")
        self.tokenized_chunks = []
        for chunk in tqdm(self.chunks):
            # Add special tokens for beginning and end
            text = f"<|startoftext|>{chunk}<|endoftext|>"
            tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            if len(tokens) > 50:  # Only keep substantial chunks
                self.tokenized_chunks.append(tokens)
        
        print(f"Created {len(self.tokenized_chunks)} tokenized training examples")
    
    def get_dataset(self):
        return Dataset.from_dict({"input_ids": self.tokenized_chunks})

def setup_model_and_tokenizer(model_name="gpt2"):
    """Load and configure model and tokenizer."""
    print(f"Loading {model_name} model and tokenizer...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {"pad_token": "<|pad|>"}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def train_model(chunks_file="dracula_chunks.txt", output_dir="stoker-style-model"):
    """Main training function."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer = setup_model_and_tokenizer()
    model.to(device)
    
    # Prepare dataset
    dataset_handler = DraculaDataset(chunks_file, tokenizer)
    train_dataset = dataset_handler.get_dataset()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Small batch size for memory efficiency
        gradient_accumulation_steps=4,   # Effective batch size = 2 * 4 = 8
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        dataloader_drop_last=True,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "base_model": "gpt2",
        "training_chunks": len(dataset_handler.tokenized_chunks),
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("Training complete!")
    return trainer

if __name__ == "__main__":
    # Check if chunks file exists
    if not os.path.exists("dracula_chunks.txt"):
        print("Error: dracula_chunks.txt not found. Please run process_dracula.py first.")
        exit(1)
    
    trainer = train_model()
    print("\nTo generate text with your trained model, run: python generate_stoker_style.py")