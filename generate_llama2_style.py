#!/usr/bin/env python3
"""
Generate text using the Llama 2 + LoRA fine-tuned model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_llama2_lora_model(
    base_model_name="meta-llama/Llama-2-7b-hf",
    lora_adapter_path="stoker-llama2-lora",
    use_auth_token=None
):
    """Load the fine-tuned Llama 2 model with LoRA adapter."""
    
    if not os.path.exists(lora_adapter_path):
        logger.error(f"LoRA adapter not found at {lora_adapter_path}")
        logger.error("Please run train_llama2_lora.py first to create the model.")
        return None, None
    
    logger.info(f"Loading base model: {base_model_name}")
    
    # Configure 4-bit quantization for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        lora_adapter_path,
        use_auth_token=use_auth_token,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=use_auth_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    
    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    logger.info("Model loaded successfully!")
    
    return model, tokenizer

def generate_stoker_text(
    model, 
    tokenizer, 
    prompt="", 
    max_length=300, 
    temperature=0.8,
    top_p=0.9,
    do_sample=True
):
    """Generate text in Stoker's style using the fine-tuned model."""
    
    device = next(model.parameters()).device
    
    # Format prompt in Llama 2 instruction format
    if prompt.strip():
        formatted_prompt = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker about: {prompt} [/INST]"
    else:
        formatted_prompt = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker: [/INST]"
    
    # Tokenize input
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
    
    # Decode and clean up
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after [/INST])
    if "[/INST]" in generated_text:
        generated_text = generated_text.split("[/INST]", 1)[1].strip()
    
    return generated_text

def interactive_mode(model, tokenizer):
    """Interactive text generation mode."""
    print("\n🧛 === Llama 2 Bram Stoker Style Generator ===")
    print("Enter a prompt (or press Enter for random generation)")
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("Gothic Prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        print("\n🏰 Generating Gothic prose...")
        try:
            generated = generate_stoker_text(
                model, tokenizer, prompt, 
                max_length=200, temperature=0.8
            )
            print(f"\n--- Generated Text ---")
            print(generated)
            print("--- End ---\n")
        except Exception as e:
            print(f"Error generating text: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate text with Llama 2 + LoRA Stoker model")
    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf", help="Base model name")
    parser.add_argument("--lora_path", default="stoker-llama2-lora", help="Path to LoRA adapter")
    parser.add_argument("--prompt", default="", help="Starting prompt for generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--auth_token", default=None, help="HuggingFace auth token")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_llama2_lora_model(
        args.base_model, 
        args.lora_path, 
        args.auth_token
    )
    
    if model is None:
        return
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        # Single generation
        print(f"🧛 Generating Gothic text with prompt: '{args.prompt}'")
        generated = generate_stoker_text(
            model, tokenizer, 
            args.prompt, 
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print("\n--- Generated Text ---")
        print(generated)
        print("--- End ---")

if __name__ == "__main__":
    main()