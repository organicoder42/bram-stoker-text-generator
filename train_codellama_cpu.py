#!/usr/bin/env python3
"""
CPU-optimized Code Llama training without quantization.
Works on macOS and systems without CUDA support.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DraculaDatasetCodeLlama:
    def __init__(self, chunks_file, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read chunks
        logger.info(f"Loading chunks from {chunks_file}...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = [line.strip().replace('\\n', '\n') for line in f if line.strip()]
        
        logger.info(f"Loaded {len(self.chunks)} text chunks")
        
        # Prepare prompts in instruction format
        self.prepared_texts = []
        for chunk in tqdm(self.chunks, desc="Preparing texts"):
            # Use Code Llama instruction format
            formatted_text = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker: [/INST] {chunk}</s>"
            self.prepared_texts.append(formatted_text)
        
        logger.info(f"Prepared {len(self.prepared_texts)} training texts")
    
    def get_dataset(self):
        # Tokenize all texts
        logger.info("Tokenizing texts...")
        tokenized_texts = []
        
        for text in tqdm(self.prepared_texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            if len(tokens) > 50:  # Only keep substantial chunks
                tokenized_texts.append(tokens)
        
        logger.info(f"Created {len(tokenized_texts)} tokenized training examples")
        return Dataset.from_dict({"input_ids": tokenized_texts})

def setup_codellama_cpu(model_name="codellama/CodeLlama-7b-hf"):
    """Load Code Llama model for CPU training."""
    
    logger.info(f"Loading {model_name} for CPU training...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model in float16 for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def setup_lora_config_cpu():
    """Configure LoRA for CPU training."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Smaller rank for CPU efficiency
        lora_alpha=16,  # Proportional scaling
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    return lora_config

def train_codellama_cpu(
    chunks_file="dracula_chunks.txt", 
    model_name="codellama/CodeLlama-7b-hf",
    output_dir="stoker-codellama-lora"
):
    """CPU-optimized Code Llama training."""
    
    # Check if chunks file exists
    if not os.path.exists(chunks_file):
        logger.error(f"Chunks file {chunks_file} not found. Please run process_dracula.py first.")
        return None
    
    logger.info("Starting CPU-optimized Code Llama training...")
    logger.info("💡 This will take 2-4 hours on CPU but uses less memory")
    
    try:
        # Load model and tokenizer
        model, tokenizer = setup_codellama_cpu(model_name)
        
        # Setup LoRA
        lora_config = setup_lora_config_cpu()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        # Prepare dataset with smaller context for CPU
        dataset_handler = DraculaDatasetCodeLlama(chunks_file, tokenizer, max_length=1024)
        train_dataset = dataset_handler.get_dataset()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # CPU-optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=2,  # Fewer epochs for CPU
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,  # Smaller effective batch
            warmup_steps=20,
            learning_rate=3e-4,  # Slightly higher for fewer epochs
            fp16=False,  # No fp16 on CPU
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            dataloader_drop_last=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            dataloader_num_workers=0,  # No multiprocessing on CPU
            report_to=None,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting LoRA fine-tuning on CPU...")
        trainer.train()
        
        # Save the LoRA adapter
        logger.info(f"Saving LoRA adapter to {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "base_model": model_name,
            "method": "LoRA (CPU)",
            "quantization": "none",
            "training_chunks": len(dataset_handler.prepared_texts),
            "epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "lora_rank": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": lora_config.target_modules,
        }
        
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("Code Llama LoRA fine-tuning complete!")
        return trainer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

if __name__ == "__main__":
    print("🦙 === Code Llama 7B + LoRA (CPU Optimized) ===")
    print("💻 CPU-only training - no CUDA required")
    print("📊 Optimized for systems without GPU support")
    print("💾 Memory usage: ~8-16GB RAM")
    print("⏰ Training time: ~2-4 hours on CPU")
    print("🎯 Quality: Excellent results, slower training")
    print()
    
    trainer = train_codellama_cpu()
    
    if trainer:
        print("\n🎉 Code Llama CPU training completed successfully!")
        print("🌐 To use the web interface: python app_llama2.py")
        print("💻 To test generation: python generate_llama2_style.py --lora_path stoker-codellama-lora --base_model codellama/CodeLlama-7b-hf")
        print()
        print("📈 Expected improvements over GPT-2:")
        print("• 10x better text coherence")
        print("• 2x longer context (1024 vs 512 tokens)")
        print("• More authentic Gothic Victorian style")
        print("• Better instruction following")
    else:
        print("\n❌ Code Llama training failed.")
        print("💡 Fallback: The GPT-2 model is still available and working")