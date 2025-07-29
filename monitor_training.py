#!/usr/bin/env python3
"""
Monitor Code Llama training progress.
"""

import os
import time
import subprocess

def check_training_status():
    """Check if training is still running and show progress."""
    
    # Check if process is running
    try:
        with open('codellama_training.pid', 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process exists
        result = subprocess.run(['ps', '-p', str(pid)], capture_output=True, text=True)
        process_running = result.returncode == 0
        
    except FileNotFoundError:
        process_running = False
        pid = None
    
    print("🦙 === Code Llama Training Monitor ===")
    print(f"📅 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if process_running:
        print(f"✅ Training process running (PID: {pid})")
    else:
        print("❌ Training process not found")
    
    # Show log file info
    if os.path.exists('codellama_training.log'):
        log_size = os.path.getsize('codellama_training.log')
        print(f"📝 Log file size: {log_size:,} bytes")
        
        # Show last few lines of log
        print("\n📋 Recent training output:")
        print("-" * 50)
        
        try:
            with open('codellama_training.log', 'r') as f:
                lines = f.readlines()
                # Show last 10 non-empty lines
                recent_lines = [line.strip() for line in lines[-20:] if line.strip()]
                for line in recent_lines[-10:]:
                    print(line)
        except Exception as e:
            print(f"Error reading log: {e}")
    else:
        print("📝 No log file found")
    
    # Check if model directory has been created
    if os.path.exists('stoker-codellama-lora'):
        files = os.listdir('stoker-codellama-lora')
        if files:
            print(f"\n🏆 Model files found: {len(files)} files in stoker-codellama-lora/")
            print("✅ Training appears to have completed successfully!")
            return True
        else:
            print(f"\n📁 Model directory exists but empty (training in progress)")
    else:
        print(f"\n📁 Model directory not yet created")
    
    return process_running

def main():
    """Main monitoring function."""
    completed = check_training_status()
    
    if completed:
        print("\n🎉 Training completed! You can now use the enhanced model:")
        print("🌐 Web interface: python app_llama2.py")
        print("💻 Command line: python generate_llama2_style.py --lora_path stoker-codellama-lora")
    else:
        print(f"\n⏰ Training still in progress. Check again in 30-60 minutes.")
        print(f"💡 Monitor with: python monitor_training.py")

if __name__ == "__main__":
    main()