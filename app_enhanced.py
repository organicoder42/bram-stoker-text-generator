#!/usr/bin/env python3
"""
Enhanced Flask web interface with model selection and improved UX.
"""

from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for loaded models
loaded_models = {}
current_model = None
device = None

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        
    def load_gpt2_model(self):
        """Load GPT-2 model."""
        try:
            gpt2_path = "stoker-style-model"
            if not os.path.exists(gpt2_path):
                return False
            
            logger.info("Loading GPT-2 model...")
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
            model = GPT2LMHeadModel.from_pretrained(gpt2_path)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            self.models['gpt2'] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': device,
                'name': 'GPT-2 Fine-tuned',
                'description': 'Fast and reliable Gothic text generation',
                'context_length': 512
            }
            
            logger.info("GPT-2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading GPT-2: {e}")
            return False
    
    def load_codellama_model(self):
        """Load Code Llama model (CPU compatible)."""
        try:
            lora_path = "stoker-codellama-lora"
            if not os.path.exists(lora_path):
                return False
            
            logger.info("Loading Code Llama model...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
            
            # Load base model without quantization for CPU
            base_model = AutoModelForCausalLM.from_pretrained(
                "codellama/CodeLlama-7b-hf",
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, lora_path)
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            model.eval()
            device = torch.device("cpu")
            
            self.models['codellama'] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': device,
                'name': 'Code Llama + LoRA',
                'description': 'High-quality Gothic prose with superior coherence',
                'context_length': 1024
            }
            
            logger.info("Code Llama model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Code Llama: {e}")
            return False
    
    def get_available_models(self):
        """Get list of available models."""
        return list(self.models.keys())
    
    def switch_model(self, model_key):
        """Switch to a different model."""
        if model_key in self.models:
            self.current_model = model_key
            logger.info(f"Switched to model: {self.models[model_key]['name']}")
            return True
        return False
    
    def get_current_model_info(self):
        """Get info about current model."""
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model]
        return None
    
    def generate_text_gpt2(self, prompt="", max_length=200, temperature=0.8):
        """Generate text using GPT-2."""
        model_info = self.models['gpt2']
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        device = model_info['device']
        
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
    
    def generate_text_codellama(self, prompt="", max_length=200, temperature=0.8):
        """Generate text using Code Llama."""
        model_info = self.models['codellama']
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Format prompt in instruction format
        if prompt.strip():
            formatted_prompt = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker about: {prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] Write a passage in the Gothic style of Bram Stoker: [/INST]"
        
        # Tokenize
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
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
    
    def generate_text(self, prompt="", max_length=200, temperature=0.8):
        """Generate text using current model."""
        if not self.current_model or self.current_model not in self.models:
            return "Error: No model selected"
        
        try:
            if self.current_model == 'gpt2':
                return self.generate_text_gpt2(prompt, max_length, temperature)
            elif self.current_model == 'codellama':
                return self.generate_text_codellama(prompt, max_length, temperature)
            else:
                return "Error: Unknown model type"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating text: {str(e)}"

# Initialize model manager
model_manager = ModelManager()

@app.route('/')
def index():
    """Main page."""
    return render_template('index_enhanced.html')

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models."""
    available_models = []
    
    for model_key, model_info in model_manager.models.items():
        available_models.append({
            'key': model_key,
            'name': model_info['name'],
            'description': model_info['description'],
            'context_length': model_info['context_length'],
            'active': model_key == model_manager.current_model
        })
    
    return jsonify({
        'models': available_models,
        'current_model': model_manager.current_model
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model."""
    try:
        data = request.get_json()
        model_key = data.get('model_key')
        
        if model_manager.switch_model(model_key):
            return jsonify({
                'success': True,
                'current_model': model_key,
                'model_name': model_manager.models[model_key]['name']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 400
            
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint for text generation."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        max_length = min(int(data.get('max_length', 200)), 500)
        temperature = max(0.1, min(float(data.get('temperature', 0.8)), 1.5))
        
        current_model_info = model_manager.get_current_model_info()
        if not current_model_info:
            return jsonify({
                'success': False,
                'error': 'No model available'
            }), 500
        
        logger.info(f"Generating with {current_model_info['name']} - prompt: '{prompt[:50]}...', length: {max_length}, temp: {temperature}")
        
        generated_text = model_manager.generate_text(prompt, max_length, temperature)
        
        return jsonify({
            'success': True,
            'text': generated_text,
            'prompt': prompt,
            'model_key': model_manager.current_model,
            'model_name': current_model_info['name']
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
    current_model_info = model_manager.get_current_model_info()
    
    return jsonify({
        'status': 'healthy' if current_model_info else 'no_model_loaded',
        'available_models': len(model_manager.models),
        'current_model': model_manager.current_model,
        'current_model_name': current_model_info['name'] if current_model_info else None
    })

def initialize_models():
    """Load available models at startup."""
    logger.info("Initializing models...")
    
    # Try to load GPT-2 first
    if model_manager.load_gpt2_model():
        model_manager.current_model = 'gpt2'
        logger.info("✅ GPT-2 set as default model")
    
    # Try to load Code Llama
    if model_manager.load_codellama_model():
        # If Code Llama loads successfully, make it the default
        model_manager.current_model = 'codellama'
        logger.info("✅ Code Llama set as default model")
    
    if not model_manager.models:
        logger.error("❌ No models could be loaded")
        return False
    
    logger.info(f"🎉 Initialized with {len(model_manager.models)} models")
    return True

if __name__ == '__main__':
    if initialize_models():
        logger.info("Starting enhanced Flask application...")
        app.run(debug=False, host='0.0.0.0', port=8080)
    else:
        logger.error("Failed to load any models. Please train models first.")
        print("Please run one of these first to create a model:")
        print("- python train_stoker_model.py (for GPT-2)")
        print("- python train_codellama_minimal.py (for Code Llama)")