#!/usr/bin/env python3
"""
Flask web interface for the Llama 2 + LoRA Bram Stoker style text generator.
"""

from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
model_type = None

def load_llama2_model(
    base_model_name="meta-llama/Llama-2-7b-hf",
    lora_adapter_path="stoker-llama2-lora",
    use_auth_token=None
):
    """Load the Llama 2 + LoRA model."""
    global model, tokenizer, device, model_type
    
    if not os.path.exists(lora_adapter_path):
        logger.error(f"LoRA adapter not found at {lora_adapter_path}")
        return False
    
    try:
        logger.info(f"Loading Llama 2 model from {lora_adapter_path}...")
        
        # Configure 4-bit quantization
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
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model_type = "llama2-lora"
        
        logger.info(f"Llama 2 model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading Llama 2 model: {e}")
        return False

def load_gpt2_fallback():
    """Fallback to GPT-2 model if Llama 2 is not available."""
    global model, tokenizer, device, model_type
    
    gpt2_path = "stoker-style-model"
    if not os.path.exists(gpt2_path):
        logger.error(f"GPT-2 model not found at {gpt2_path}")
        return False
    
    try:
        logger.info("Loading GPT-2 fallback model...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
        model = GPT2LMHeadModel.from_pretrained(gpt2_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        model_type = "gpt2"
        
        logger.info(f"GPT-2 model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading GPT-2 model: {e}")
        return False

def generate_text_llama2(prompt="", max_length=200, temperature=0.8):
    """Generate text using Llama 2 model."""
    if model is None or tokenizer is None:
        return "Error: Model not loaded"
    
    try:
        device = next(model.parameters()).device
        
        # Format prompt in Llama 2 instruction format
        if prompt.strip():
            formatted_prompt = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker about: {prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker: [/INST]"
        
        # Tokenize
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
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
        
    except Exception as e:
        logger.error(f"Error generating text with Llama 2: {e}")
        return f"Error generating text: {str(e)}"

def generate_text_gpt2(prompt="", max_length=200, temperature=0.8):
    """Generate text using GPT-2 model."""
    if model is None or tokenizer is None:
        return "Error: Model not loaded"
    
    try:
        device = next(model.parameters()).device
        
        # Encode the prompt
        if prompt.strip():
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
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode("<|endoftext|>")[0]
            )
        
        # Decode and clean up
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output if it was included
        if prompt.strip() and text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        # Clean up any remaining special tokens
        text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error generating text with GPT-2: {e}")
        return f"Error generating text: {str(e)}"

def generate_text(prompt="", max_length=200, temperature=0.8):
    """Generate text using the loaded model."""
    if model_type == "llama2-lora":
        return generate_text_llama2(prompt, max_length, temperature)
    else:
        return generate_text_gpt2(prompt, max_length, temperature)

@app.route('/')
def index():
    """Main page."""
    return render_template('index_llama2.html')

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint for text generation."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        max_length = min(int(data.get('max_length', 200)), 500)  # Cap at 500
        temperature = max(0.1, min(float(data.get('temperature', 0.8)), 1.5))  # Between 0.1 and 1.5
        
        logger.info(f"Generating text with {model_type} - prompt: '{prompt[:50]}...', length: {max_length}, temp: {temperature}")
        
        generated_text = generate_text(prompt, max_length, temperature)
        
        return jsonify({
            'success': True,
            'text': generated_text,
            'prompt': prompt,
            'model_type': model_type
        })
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    model_loaded = model is not None and tokenizer is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'model_type': model_type,
        'device': str(device) if device else None
    })

def load_best_available_model():
    """Load the best available model (Llama 2 first, then GPT-2 fallback)."""
    
    # Try Llama 2 first
    if load_llama2_model():
        logger.info("✅ Llama 2 + LoRA model loaded successfully")
        return True
    
    # Try Code Llama alternative
    if load_llama2_model(
        base_model_name="codellama/CodeLlama-7b-hf",
        lora_adapter_path="stoker-codellama-lora"
    ):
        logger.info("✅ Code Llama + LoRA model loaded successfully")
        return True
    
    # Fallback to GPT-2
    if load_gpt2_fallback():
        logger.info("✅ GPT-2 model loaded as fallback")
        return True
    
    logger.error("❌ No models could be loaded")
    return False

if __name__ == '__main__':
    # Load best available model at startup
    if load_best_available_model():
        logger.info("Starting Flask application...")
        app.run(debug=False, host='0.0.0.0', port=8080)
    else:
        logger.error("Failed to load any model. Please train a model first.")
        print("Please run one of these first to create a model:")
        print("- python train_llama2_lora.py (for Llama 2)")
        print("- python train_stoker_model.py (for GPT-2)")
        print("- python train_codellama_alternative.py (for Code Llama)")