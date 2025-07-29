#!/bin/bash

echo "🧛 Starting Bram Stoker Text Generator Web Interface"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "dracula_env" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Check if model exists
if [ ! -d "stoker-style-model" ]; then
    echo "❌ Model not found. Please train the model first:"
    echo "   python train_stoker_model.py"
    exit 1
fi

echo "✅ Model found"
echo "🔄 Activating virtual environment..."

# Activate virtual environment and start server
source dracula_env/bin/activate

echo "🚀 Starting web server on port 8080..."
echo "📍 Open your browser to: http://localhost:8080"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python run_web_app.py