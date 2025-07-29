#!/usr/bin/env python3
"""
Generate text in Bram Stoker's style using the fine-tuned model.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import os

def load_model(model_path="stoker-style-model"):
    """Load the fine-tuned model and tokenizer."""
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' not found.")
        print("Please run train_stoker_model.py first to create the model.")
        return None, None
    
    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt="", max_length=300, temperature=0.8, num_return_sequences=1):
    """Generate text in Stoker's style."""
    device = next(model.parameters()).device
    
    # Encode the prompt
    if prompt:
        input_text = f"<|startoftext|>{prompt}"
    else:
        input_text = "<|startoftext|>"
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|endoftext|>")[0]
        )
    
    # Decode and clean up
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Remove the prompt from the output if it was included
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()
        generated_texts.append(text)
    
    return generated_texts

def interactive_mode(model, tokenizer):
    """Interactive text generation mode."""
    print("\n=== Bram Stoker Style Text Generator ===")
    print("Enter a prompt (or press Enter for random generation)")
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        print("\nGenerating...")
        try:
            generated = generate_text(model, tokenizer, prompt, max_length=200)
            print(f"\n--- Generated Text ---")
            print(generated[0])
            print("--- End ---\n")
        except Exception as e:
            print(f"Error generating text: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate text in Bram Stoker's style")
    parser.add_argument("--model_path", default="stoker-style-model", help="Path to the fine-tuned model")
    parser.add_argument("--prompt", default="", help="Starting prompt for generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1=conservative, 1.0=creative)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    if model is None:
        return
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        # Single generation
        print(f"Generating text with prompt: '{args.prompt}'")
        generated = generate_text(
            model, tokenizer, 
            args.prompt, 
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print("\n--- Generated Text ---")
        print(generated[0])
        print("--- End ---")

if __name__ == "__main__":
    main()