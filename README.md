# 🧛 Bram Stoker Style Text Generator

Generate Gothic prose in the style of Bram Stoker using a fine-tuned GPT-2 model trained on *Dracula*. Features a beautiful Victorian-themed web interface for interactive text generation.

![Demo](https://img.shields.io/badge/status-working-brightgreen) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- 🎭 **Gothic Text Generation**: Creates atmospheric prose in Bram Stoker's distinctive style
- 🌐 **Web Interface**: Beautiful Victorian-themed UI with dark Gothic styling
- ⚙️ **Customizable Controls**: Adjust text length (50-400 words) and creativity (0.3-1.2)
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- 📋 **Copy Function**: One-click copying of generated text
- 🔄 **Real-time Generation**: Interactive text generation with loading states

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/dracula-model.git
cd dracula-model

# Create virtual environment
python3 -m venv dracula_env
source dracula_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Process Training Data
```bash
# Download and process Dracula text from Project Gutenberg
python process_dracula.py
```

### 3. Train the Model
```bash
# Fine-tune GPT-2 on Stoker's writing style (20-60 minutes)
python train_stoker_model.py
```

### 4. Launch Web Interface
```bash
# Start the Gothic web interface
./start_server.sh
# Or manually:
python run_web_app.py
```

### 5. Generate Text
Open your browser to **http://localhost:8080** and start creating Gothic prose!

## 📁 Project Structure

```
dracula-model/
├── 📄 README.md                 # This file
├── ⚙️ requirements.txt          # Python dependencies
├── 🐍 process_dracula.py        # Text processing and chunking
├── 🧠 train_stoker_model.py     # Model training script
├── 🌐 app.py                    # Flask web application
├── 🚀 run_web_app.py           # Server startup script
├── 📜 start_server.sh          # Bash startup script
├── 🎭 generate_stoker_style.py # Command-line text generation
├── 📚 WEB_INTERFACE.md         # Web interface documentation
├── 🗂️ templates/
│   └── 🎨 index.html           # Gothic web interface
├── 🎨 static/
│   └── 💅 style.css           # Victorian Gothic styling
└── 🏰 stoker-style-model/      # Trained model (created after training)
```

## 🎯 Usage Examples

### Web Interface
- Enter a prompt like "The ancient castle stood upon..."
- Adjust creativity slider (lower = conservative, higher = creative)
- Set desired text length
- Click "Summon the Words" to generate Gothic prose

### Command Line
```bash
# Interactive mode
python generate_stoker_style.py --interactive

# Single generation
python generate_stoker_style.py --prompt "The moonlight cast eerie shadows"

# Custom parameters
python generate_stoker_style.py --prompt "My dear Mina," --temperature 0.9 --max_length 300
```

### API Usage
```bash
# Health check
curl http://localhost:8080/health

# Generate text
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The castle loomed", "max_length": 200, "temperature": 0.8}'
```

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: GPT-2 (117M parameters)
- **Training Data**: Bram Stoker's *Dracula* (161K words, 179 chunks)
- **Fine-tuning**: 3 epochs with learning rate 5e-5
- **Context Length**: 512 tokens with 100-word overlap

### Training Specifications
- **Hardware**: CPU/GPU compatible (CUDA optional)
- **Memory**: ~4GB VRAM for GPU training, ~2GB RAM minimum
- **Time**: 20-30 minutes (GPU) / 2-4 hours (CPU)

### Web Interface
- **Framework**: Flask with Jinja2 templating
- **Styling**: Custom Gothic CSS with Google Fonts
- **JavaScript**: Vanilla JS for interactive controls
- **Responsive**: Mobile-first design approach

## 🎨 What the Model Learns

The fine-tuned model captures:
- **Victorian Gothic Atmosphere**: Dark, mysterious, supernatural elements
- **Epistolary Format**: Diary entries and letter-style narratives  
- **Period Vocabulary**: 19th-century language patterns and terminology
- **Character Voices**: Distinctive speaking styles from the novel
- **Narrative Structure**: First-person journal entries and dramatic descriptions

## 📊 Performance

- **Generation Speed**: ~2-3 seconds per 200 words (CPU)
- **Quality**: Coherent Gothic prose with proper Victorian style
- **Diversity**: Temperature control allows conservative to creative output
- **Consistency**: Maintains Stoker's voice across different prompts

## 🔧 Configuration

### Model Parameters
Edit `train_stoker_model.py`:
- `num_train_epochs`: Training epochs (default: 3)
- `learning_rate`: Learning rate (default: 5e-5)
- `max_length`: Token limit (default: 512)

### Web Interface
Edit `app.py`:
- `port`: Server port (default: 8080)
- `max_length`: Generation limit (default: 400)
- `temperature`: Default creativity (default: 0.8)

## 🚨 Requirements

- **Python**: 3.8+ required
- **Memory**: 4GB+ RAM recommended
- **Storage**: ~2GB for model and dependencies
- **Network**: Internet connection for initial setup

## 📚 Dependencies

Core libraries:
- `torch>=2.0.0` - PyTorch for neural networks
- `transformers>=4.30.0` - Hugging Face model library
- `flask>=2.3.0` - Web framework
- `datasets>=2.12.0` - Data processing
- `numpy>=1.21.0` - Numerical computing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bram Stoker** - Original author of *Dracula*
- **Project Gutenberg** - Public domain text source
- **Hugging Face** - Transformers library and model hub
- **OpenAI** - GPT-2 base model architecture

## 📈 Roadmap

- [ ] Support for other Gothic authors (Poe, Shelley, etc.)
- [ ] Longer context models (GPT-2 Medium/Large)
- [ ] Character-specific fine-tuning
- [ ] Advanced web interface features
- [ ] Docker containerization
- [ ] REST API documentation

---

*"There are darknesses in life and there are lights, and you are one of the lights, the light of all lights."* - Bram Stoker, Dracula