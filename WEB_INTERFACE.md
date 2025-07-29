# 🧛 Bram Stoker Web Interface

A Gothic-themed web interface for generating text in the style of Bram Stoker.

## Quick Start

1. **Ensure model is trained:**
```bash
python train_stoker_model.py
```

2. **Install Flask (if not already installed):**
```bash
source dracula_env/bin/activate
pip install flask
```

3. **Start the web server:**
```bash
source dracula_env/bin/activate
python run_web_app.py
```

4. **Open your browser:**
Visit `http://localhost:5000`

## Features

### 🎭 User Interface
- **Gothic Victorian Theme**: Dark, atmospheric styling with crimson and gold accents
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Controls**: Adjust text length and creativity with sliders
- **Copy Function**: One-click copying of generated text

### ⚙️ Generation Controls
- **Prompt Input**: Start your Gothic tale or leave empty for random generation
- **Length Control**: 50-400 words (adjustable slider)
- **Temperature Control**: 0.3-1.2 creativity level
  - Lower = More conservative, closer to training text
  - Higher = More creative and varied output

### 🔧 Technical Features
- **Health Check**: Automatic model status verification
- **Error Handling**: Graceful handling of generation errors
- **Loading States**: Visual feedback during text generation
- **Keyboard Shortcuts**: Ctrl+Enter or Cmd+Enter to generate

## API Endpoints

### `POST /generate`
Generate text with custom parameters.

**Request:**
```json
{
  "prompt": "The ancient castle stood upon",
  "max_length": 200,
  "temperature": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "text": "Generated Gothic prose...",
  "prompt": "The ancient castle stood upon"
}
```

### `GET /health`
Check model status and server health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

## Files Structure

```
DraculaModel/
├── app.py                 # Flask application
├── run_web_app.py        # Startup script
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── style.css         # Gothic styling
└── stoker-style-model/   # Trained model directory
```

## Customization

### Styling
Edit `static/style.css` to customize the Gothic theme:
- Colors: Modify CSS variables in `:root`
- Fonts: Change Google Fonts imports
- Layout: Adjust grid and flexbox properties

### Model Parameters
Edit `app.py` to change default generation settings:
- `max_length`: Maximum text length limit
- `temperature`: Default creativity level
- Model loading parameters

## Troubleshooting

### Model Not Found
```
❌ Error: Model not found!
```
**Solution**: Run `python train_stoker_model.py` first.

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill existing process or change port in `app.py`.

### Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: Model runs on CPU by default. Reduce batch size if needed.

## Production Deployment

For production use:
1. Set `debug=False` in `app.py`
2. Use a production WSGI server like Gunicorn
3. Configure proper logging
4. Add authentication if needed
5. Set up reverse proxy (nginx/Apache)

## Browser Compatibility

- ✅ Chrome/Chromium 80+
- ✅ Firefox 75+  
- ✅ Safari 13+
- ✅ Edge 80+
- ⚠️ Internet Explorer not supported