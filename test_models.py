#!/usr/bin/env python3
"""
Test both GPT-2 and Code Llama models to compare performance.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import time

def test_gpt2_model():
    """Test the original GPT-2 model."""
    print("🤖 Testing GPT-2 Model...")
    
    try:
        gpt2_path = "stoker-style-model"
        if not os.path.exists(gpt2_path):
            print("❌ GPT-2 model not found")
            return None
        
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
        model = GPT2LMHeadModel.from_pretrained(gpt2_path)
        model.eval()
        
        # Test prompt
        prompt = "The ancient castle stood upon the hill"
        input_text = f"<|startoftext|>{prompt}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=150,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode("<|endoftext|>")[0]
            )
        generation_time = time.time() - start_time
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()
        
        return {
            "model": "GPT-2 Fine-tuned",
            "text": text,
            "time": generation_time,
            "length": len(text.split())
        }
        
    except Exception as e:
        print(f"❌ GPT-2 test failed: {e}")
        return None

def test_codellama_cpu():
    """Test Code Llama model without quantization."""
    print("🦙 Testing Code Llama Model (CPU)...")
    
    try:
        lora_path = "stoker-codellama-lora"
        if not os.path.exists(lora_path):
            print("❌ Code Llama LoRA adapter not found")
            return None
        
        # Load tokenizer from LoRA directory
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        
        # Load base model without quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Test prompt in instruction format
        prompt = "The ancient castle stood upon the hill"
        formatted_prompt = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker about: {prompt} [/INST]"
        
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + 150,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
        generation_time = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (after [/INST])
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]", 1)[1].strip()
        
        return {
            "model": "Code Llama + LoRA",
            "text": generated_text,
            "time": generation_time,
            "length": len(generated_text.split())
        }
        
    except Exception as e:
        print(f"❌ Code Llama test failed: {e}")
        return None

def compare_models():
    """Compare both models side by side."""
    print("🧛 === Bram Stoker Model Comparison ===\n")
    
    # Test both models
    gpt2_result = test_gpt2_model()
    print()
    codellama_result = test_codellama_cpu()
    print()
    
    # Display results
    print("📊 === COMPARISON RESULTS ===\n")
    
    if gpt2_result:
        print("🤖 **GPT-2 Fine-tuned Model:**")
        print(f"⏰ Generation Time: {gpt2_result['time']:.1f} seconds")
        print(f"📝 Text Length: {gpt2_result['length']} words")
        print("📖 Generated Text:")
        print(f'"{gpt2_result["text"][:300]}..."')
        print()
    
    if codellama_result:
        print("🦙 **Code Llama + LoRA Model:**")
        print(f"⏰ Generation Time: {codellama_result['time']:.1f} seconds")
        print(f"📝 Text Length: {codellama_result['length']} words")
        print("📖 Generated Text:")
        print(f'"{codellama_result["text"][:300]}..."')
        print()
    
    # Analysis
    if gpt2_result and codellama_result:
        print("🔍 **Analysis:**")
        
        if codellama_result['length'] > gpt2_result['length']:
            print("✅ Code Llama generates longer, more detailed text")
        
        if codellama_result['time'] > gpt2_result['time']:
            print("⚠️ Code Llama is slower but more sophisticated")
            
        print("📈 **Expected Code Llama Advantages:**")
        print("• More coherent narrative flow")
        print("• Better Gothic vocabulary and atmosphere")
        print("• Longer context understanding")
        print("• More authentic Victorian style")
        
    elif gpt2_result:
        print("✅ GPT-2 model is working perfectly")
        print("💡 Code Llama model needs troubleshooting")
        
    elif codellama_result:
        print("✅ Code Llama model is working perfectly")
        print("💡 This is a significant upgrade over GPT-2!")
    
    else:
        print("❌ Both models need troubleshooting")

if __name__ == "__main__":
    compare_models()