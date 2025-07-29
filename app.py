#!/usr/bin/env python3
"""
Flask web interface for the Bram Stoker style text generator.
"""

from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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

def load_model(model_path="stoker-style-model"):
    """Load the fine-tuned model and tokenizer."""
    global model, tokenizer, device
    
    if not os.path.exists(model_path):
        logger.error(f"Model directory '{model_path}' not found.")
        return False
    
    try:
        logger.info(f"Loading model from {model_path}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def generate_text(prompt="", max_length=200, temperature=0.8):
    """Generate text in Stoker's style."""
    if model is None or tokenizer is None:
        return "Error: Model not loaded"
    
    try:
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
        logger.error(f"Error generating text: {e}")
        return f"Error generating text: {str(e)}"

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint for text generation."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        max_length = min(int(data.get('max_length', 200)), 500)  # Cap at 500
        temperature = max(0.1, min(float(data.get('temperature', 0.8)), 1.5))  # Between 0.1 and 1.5
        
        logger.info(f"Generating text with prompt: '{prompt[:50]}...', length: {max_length}, temp: {temperature}")
        
        generated_text = generate_text(prompt, max_length, temperature)
        
        return jsonify({
            'success': True,
            'text': generated_text,
            'prompt': prompt
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
        'device': str(device) if device else None
    })

if __name__ == '__main__':
    # Load model at startup
    if load_model():
        logger.info("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        logger.error("Failed to load model. Please ensure the model has been trained first.")
        print("Please run 'python train_stoker_model.py' first to create the model.")