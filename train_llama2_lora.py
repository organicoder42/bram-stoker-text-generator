#!/usr/bin/env python3
"""
Fine-tune Llama 2 7B on Bram Stoker's writing style using LoRA and 4-bit quantization.
This provides significantly better performance than GPT-2 while being memory-efficient.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DraculaDatasetLlama:
    def __init__(self, chunks_file, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read chunks
        logger.info(f"Loading chunks from {chunks_file}...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = [line.strip().replace('\\n', '\n') for line in f if line.strip()]
        
        logger.info(f"Loaded {len(self.chunks)} text chunks")
        
        # Prepare prompts in Llama 2 chat format for better results
        self.prepared_texts = []
        for chunk in tqdm(self.chunks, desc="Preparing texts"):
            # Use Llama 2 instruction format for better fine-tuning
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

def setup_llama2_model(model_name="meta-llama/Llama-2-7b-hf", use_auth_token=None):
    """Load and configure Llama 2 model with 4-bit quantization."""
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    
    logger.info(f"Loading {model_name} with 4-bit quantization...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=use_auth_token,
        trust_remote_code=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=use_auth_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA parameters for efficient fine-tuning."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],  # Target all attention and MLP layers
        bias="none",
    )
    return lora_config

def train_llama2_model(
    chunks_file="dracula_chunks.txt", 
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="stoker-llama2-lora",
    use_auth_token=None
):
    """Main training function for Llama 2 + LoRA."""
    
    # Check if chunks file exists
    if not os.path.exists(chunks_file):
        logger.error(f"Chunks file {chunks_file} not found. Please run process_dracula.py first.")
        return None
    
    # Setup device and memory optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load model and tokenizer
        model, tokenizer = setup_llama2_model(model_name, use_auth_token)
        
        # Setup LoRA
        lora_config = setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        # Prepare dataset
        dataset_handler = DraculaDatasetLlama(chunks_file, tokenizer, max_length=2048)
        train_dataset = dataset_handler.get_dataset()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Training arguments optimized for LoRA
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Small batch size for memory efficiency
            gradient_accumulation_steps=8,   # Effective batch size = 1 * 8 = 8
            warmup_steps=50,
            learning_rate=2e-4,  # Higher learning rate for LoRA
            fp16=not torch.cuda.is_available(),  # Use fp16 on CPU, bf16 on CUDA
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            dataloader_drop_last=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            group_by_length=True,
            report_to=None,  # Disable wandb/tensorboard
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
        logger.info("Starting LoRA fine-tuning...")
        trainer.train()
        
        # Save the LoRA adapter
        logger.info(f"Saving LoRA adapter to {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "base_model": model_name,
            "method": "LoRA",
            "quantization": "4-bit",
            "training_chunks": len(dataset_handler.prepared_texts),
            "epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "lora_rank": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": lora_config.target_modules,
        }
        
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("LoRA fine-tuning complete!")
        return trainer
        
    except Exception as e:
        if "requires you to log in" in str(e) or "gated repository" in str(e):
            logger.error(f"""
            Error: {e}
            
            Llama 2 is a gated model. You need to:
            1. Go to https://huggingface.co/meta-llama/Llama-2-7b-hf and request access
            2. Accept the license terms
            3. Login with: huggingface-cli login
            4. Or pass your HF token: use_auth_token='your_token_here'
            """)
            return None
        else:
            logger.error(f"Training failed: {e}")
            return None

def create_fallback_script():
    """Create a script to use Code Llama instead if Llama 2 is not accessible."""
    fallback_script = '''#!/usr/bin/env python3
"""
Fallback script using Code Llama 7B which doesn't require gated access.
Code Llama is based on Llama 2 and works well for text generation.
"""

def train_with_code_llama():
    from train_llama2_lora import train_llama2_model
    
    print("🦙 Using Code Llama 7B (no gated access required)")
    print("This model is based on Llama 2 and works excellently for text generation.")
    
    trainer = train_llama2_model(
        model_name="codellama/CodeLlama-7b-hf",
        output_dir="stoker-codellama-lora"
    )
    
    if trainer:
        print("✅ Code Llama training completed successfully!")
    else:
        print("❌ Training failed.")

if __name__ == "__main__":
    train_with_code_llama()
'''
    
    with open("train_codellama_alternative.py", "w") as f:
        f.write(fallback_script)
    
    logger.info("Created fallback script: train_codellama_alternative.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Llama 2 7B with LoRA on Stoker's style")
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--output_dir", default="stoker-llama2-lora", help="Output directory")
    parser.add_argument("--auth_token", default=None, help="HuggingFace auth token")
    parser.add_argument("--chunks_file", default="dracula_chunks.txt", help="Training data file")
    
    args = parser.parse_args()
    
    # Create fallback script
    create_fallback_script()
    
    print("🧛 Starting Llama 2 7B + LoRA fine-tuning...")
    print("📊 This will use 4-bit quantization and LoRA for memory efficiency")
    print("💾 Expected memory usage: ~8-12GB GPU / ~16GB CPU")
    print("⏰ Training time: ~30-60 minutes on GPU, 2-4 hours on CPU")
    print()
    
    trainer = train_llama2_model(
        chunks_file=args.chunks_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_auth_token=args.auth_token
    )
    
    if trainer is None:
        print("\n💡 Alternative: Try running train_codellama_alternative.py")
        print("   Code Llama is based on Llama 2 but doesn't require gated access")