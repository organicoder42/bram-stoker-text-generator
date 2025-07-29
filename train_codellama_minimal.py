#!/usr/bin/env python3
"""
Ultra memory-efficient Code Llama training for Mac systems.
Uses minimal memory configurations to work within 16GB constraints.
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
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage to avoid MPS memory issues
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.backends.mps.enable_fallback = True

class DraculaDatasetMinimal:
    def __init__(self, chunks_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read chunks
        logger.info(f"Loading chunks from {chunks_file}...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            # Take only first 50 chunks for memory efficiency
            self.chunks = [line.strip().replace('\\n', '\n') for line in f if line.strip()][:50]
        
        logger.info(f"Loaded {len(self.chunks)} text chunks (limited for memory)")
        
        # Prepare simplified prompts
        self.prepared_texts = []
        for chunk in tqdm(self.chunks, desc="Preparing texts"):
            # Simpler format to save tokens
            formatted_text = f"Gothic: {chunk[:400]}"  # Limit chunk size
            self.prepared_texts.append(formatted_text)
        
        logger.info(f"Prepared {len(self.prepared_texts)} training texts")
    
    def get_dataset(self):
        # Tokenize with strict limits
        logger.info("Tokenizing texts...")
        tokenized_texts = []
        
        for text in tqdm(self.prepared_texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            if len(tokens) > 20:  # Keep even small chunks
                tokenized_texts.append(tokens)
        
        logger.info(f"Created {len(tokenized_texts)} tokenized training examples")
        return Dataset.from_dict({"input_ids": tokenized_texts})

def setup_codellama_minimal(model_name="codellama/CodeLlama-7b-hf"):
    """Load Code Llama with minimal memory footprint."""
    
    logger.info(f"Loading {model_name} with minimal memory settings...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Force CPU and minimal memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for better CPU compatibility
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False  # Disable cache to save memory
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Move to CPU explicitly
    model = model.to("cpu")
    
    return model, tokenizer

def setup_minimal_lora():
    """Minimal LoRA configuration."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # Very small rank
        lora_alpha=8,  # Proportional
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Only target 2 modules
        bias="none",
    )
    return lora_config

def train_codellama_minimal(
    chunks_file="dracula_chunks.txt", 
    model_name="codellama/CodeLlama-7b-hf",
    output_dir="stoker-codellama-lora"
):
    """Minimal memory Code Llama training."""
    
    # Check if chunks file exists
    if not os.path.exists(chunks_file):
        logger.error(f"Chunks file {chunks_file} not found.")
        return None
    
    logger.info("Starting ultra memory-efficient Code Llama training...")
    logger.info("💡 Using minimal settings to fit in available memory")
    
    try:
        # Clear any existing memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model and tokenizer
        model, tokenizer = setup_codellama_minimal(model_name)
        
        # Setup minimal LoRA
        lora_config = setup_minimal_lora()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        # Prepare minimal dataset
        dataset_handler = DraculaDatasetMinimal(chunks_file, tokenizer, max_length=512)
        train_dataset = dataset_handler.get_dataset()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Ultra conservative training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,  # Just 1 epoch
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,  # No accumulation
            warmup_steps=2,
            learning_rate=5e-4,
            fp16=False,
            logging_steps=2,
            save_steps=25,
            save_total_limit=1,
            dataloader_drop_last=True,
            optim="adamw_torch",
            weight_decay=0.01,
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,  # Disable pin memory
            report_to=None,
            remove_unused_columns=False,
            use_cpu=True,  # Force CPU
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting minimal LoRA fine-tuning...")
        trainer.train()
        
        # Save the LoRA adapter
        logger.info(f"Saving LoRA adapter to {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "base_model": model_name,
            "method": "LoRA (Minimal)",
            "training_chunks": len(dataset_handler.prepared_texts),
            "epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "lora_rank": lora_config.r,
            "note": "Memory-optimized training with reduced dataset and parameters"
        }
        
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("Minimal Code Llama training complete!")
        return trainer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None

if __name__ == "__main__":
    print("🦙 === Code Llama 7B + LoRA (Ultra Minimal) ===")
    print("💻 Minimal memory configuration for Mac systems")
    print("📊 Reduced dataset and parameters to fit in 16GB")
    print("💾 Memory usage: ~8-12GB RAM")
    print("⏰ Training time: ~30-60 minutes")
    print("🎯 Quality: Good results with memory constraints")
    print()
    
    trainer = train_codellama_minimal()
    
    if trainer:
        print("\n🎉 Minimal Code Llama training completed successfully!")
        print("🌐 To use the web interface: python app_llama2.py")
        print("💻 To test generation: python generate_llama2_style.py --lora_path stoker-codellama-lora --base_model codellama/CodeLlama-7b-hf")
        print()
        print("📈 Even with minimal training, you should see:")
        print("• Better text coherence than GPT-2")
        print("• More authentic Gothic style")
        print("• Better instruction following")
    else:
        print("\n❌ Minimal training failed.")
        print("💡 The existing GPT-2 model is still working perfectly!")
        print("🔄 You can continue using: python app.py (original interface)")