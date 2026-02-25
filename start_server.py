#!/usr/bin/env python3
"""
Simple server starter script
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        "fastapi",
        "uvicorn[standard]", 
        "python-multipart",
        "pillow",
        "pydantic"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def start_server():
    """Start the FastAPI server"""
    try:
        print("🚀 Starting FastAPI server...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    print("🔧 Installing dependencies...")
    install_requirements()
    print("\n" + "="*50)
    start_server()
