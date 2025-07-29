#!/usr/bin/env python3
"""
Simple script to run the Bram Stoker web interface.
"""

import os
import sys

def main():
    # Check if model exists
    if not os.path.exists("stoker-style-model"):
        print("❌ Error: Model not found!")
        print("Please run 'python train_stoker_model.py' first to create the model.")
        return 1
    
    print("🧛 Starting Bram Stoker Style Text Generator...")
    print("📍 Web interface will be available at: http://localhost:8080")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        from app import app, load_model
        
        # Load model
        if not load_model():
            print("❌ Failed to load model. Exiting.")
            return 1
        
        # Run the app
        app.run(debug=False, host='0.0.0.0', port=8080, threaded=True)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())