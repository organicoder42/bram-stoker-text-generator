#!/usr/bin/env python3
"""
Master training script - Choose your model architecture for the Bram Stoker text generator.
"""

import os
import sys
import argparse

def show_model_options():
    """Display available model options."""
    print("🧛 === Bram Stoker Text Generator - Model Training ===\n")
    
    print("📊 Available Model Options:\n")
    
    print("1. 🤖 GPT-2 Fine-tuned (Original)")
    print("   • Model: GPT-2 117M parameters")
    print("   • Method: Full fine-tuning")
    print("   • Memory: 2-4GB")
    print("   • Training: 20-60 minutes")
    print("   • Quality: Good baseline")
    print("   • Script: python train_stoker_model.py")
    print()
    
    print("2. 🦙 Llama 2 7B + LoRA (Best Quality - Gated)")
    print("   • Model: Llama 2 7B parameters")
    print("   • Method: LoRA + 4-bit quantization")
    print("   • Memory: 8-12GB")
    print("   • Training: 30-90 minutes")
    print("   • Quality: Excellent")
    print("   • Requires: Hugging Face access approval")
    print("   • Script: python train_llama2_lora.py")
    print()
    
    print("3. 🦙 Code Llama 7B + LoRA (Recommended)")
    print("   • Model: Code Llama 7B (based on Llama 2)")
    print("   • Method: LoRA + 4-bit quantization")
    print("   • Memory: 8-12GB")
    print("   • Training: 30-90 minutes")
    print("   • Quality: Excellent (95% of Llama 2)")
    print("   • Requires: No gated access")
    print("   • Script: python train_codellama_alternative.py")
    print()

def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking Prerequisites:\n")
    
    # Check chunks file
    if os.path.exists("dracula_chunks.txt"):
        with open("dracula_chunks.txt", 'r') as f:
            chunks = len([line for line in f if line.strip()])
        print(f"✅ Training data: {chunks} chunks ready")
    else:
        print("❌ Training data missing")
        print("   Run: python process_dracula.py")
        return False
    
    # Check dependencies
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    # Check advanced dependencies
    try:
        import peft
        print(f"✅ PEFT (LoRA): {peft.__version__}")
        advanced_available = True
    except ImportError:
        print("⚠️  PEFT not found (needed for Llama models)")
        advanced_available = False
    
    try:
        import bitsandbytes
        print(f"✅ BitsAndBytes: {bitsandbytes.__version__}")
    except ImportError:
        print("⚠️  BitsAndBytes not found (needed for quantization)")
        advanced_available = False
    
    print()
    
    if advanced_available:
        print("🚀 All dependencies ready - You can use any model!")
    else:
        print("📋 Only GPT-2 available - Install missing deps for Llama models:")
        print("   pip install peft bitsandbytes")
    
    return True

def run_training(model_choice):
    """Run training based on user choice."""
    
    if model_choice == "1" or model_choice.lower() == "gpt2":
        print("\n🤖 Starting GPT-2 training...")
        os.system("python train_stoker_model.py")
        
    elif model_choice == "2" or model_choice.lower() == "llama2":
        print("\n🦙 Starting Llama 2 training...")
        print("⚠️  Note: Requires Hugging Face access to meta-llama/Llama-2-7b-hf")
        print("   Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print("   Then: huggingface-cli login")
        print()
        os.system("python train_llama2_lora.py")
        
    elif model_choice == "3" or model_choice.lower() == "codellama":
        print("\n🦙 Starting Code Llama training...")
        os.system("python train_codellama_alternative.py")
        
    else:
        print("❌ Invalid choice. Please select 1, 2, or 3.")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Train Bram Stoker text generation models")
    parser.add_argument("--model", choices=["gpt2", "llama2", "codellama"], 
                       help="Model to train directly")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--check", action="store_true", help="Check prerequisites only")
    
    args = parser.parse_args()
    
    if args.list:
        show_model_options()
        return
    
    if args.check:
        check_prerequisites()
        return
    
    if args.model:
        if not check_prerequisites():
            sys.exit(1)
        
        model_map = {"gpt2": "1", "llama2": "2", "codellama": "3"}
        run_training(model_map[args.model])
        return
    
    # Interactive mode
    show_model_options()
    
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    print("🎯 Recommendation: Option 3 (Code Llama) for best quality without access requirements\n")
    
    while True:
        choice = input("Select model to train (1-3, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        if choice in ["1", "2", "3"]:
            if run_training(choice):
                print("\n🎉 Training completed!")
                print("🌐 Launch web interface: python app_llama2.py")
                break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 'q'.")

if __name__ == "__main__":
    main()