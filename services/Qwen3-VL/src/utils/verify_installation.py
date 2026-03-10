#!/usr/bin/env python3
"""
Verify installation and dependencies for Qwen3-VL API
"""

import sys
import subprocess
from pathlib import Path

def check_command(cmd):
    """Check if a command is available."""
    try:
        subprocess.run([cmd, '--version'], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False

def check_python_package(package):
    """Check if a Python package is installed."""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        return False

def main():
    print("="*80)
    print("Qwen3-VL API Installation Verification")
    print("="*80)
    
    all_ok = True
    
    # Check Python version
    print("\n[1] Checking Python version...")
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"  ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ✗ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
        all_ok = False
    
    # Check required system commands
    print("\n[2] Checking system dependencies...")
    
    if check_command('pdftoppm'):
        print("  ✓ poppler-utils installed")
    else:
        print("  ✗ poppler-utils not found")
        print("    Install with: sudo apt-get install -y poppler-utils")
        all_ok = False
    
    # Check Python packages
    print("\n[3] Checking Python packages...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'vllm',
        'PIL',  # Pillow
        'pdf2image',
        'pydantic',
    ]
    
    for package in required_packages:
        package_name = 'Pillow' if package == 'PIL' else package
        if check_python_package(package):
            print(f"  ✓ {package_name}")
        else:
            print(f"  ✗ {package_name} not installed")
            all_ok = False
    
    # Check files exist
    print("\n[4] Checking API files...")
    
    required_files = [
        'api_vllm.py',
        'requirements_api.txt',
        'start_api.sh',
        'test_client.py',
    ]
    
    for filename in required_files:
        filepath = Path(__file__).parent / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found")
            all_ok = False
    
    # Check CUDA availability
    print("\n[5] Checking GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {gpu_count} GPU(s)")
            print(f"    Device: {gpu_name}")
        else:
            print("  ⚠ CUDA not available (CPU mode will be very slow)")
            print("    Consider using a GPU for better performance")
    except ImportError:
        print("  ✗ PyTorch not installed")
        all_ok = False
    
    # Summary
    print("\n" + "="*80)
    if all_ok:
        print("✓ All checks passed! You can start the API with:")
        print("  ./start_api.sh")
        print("  or")
        print("  python api_vllm.py")
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements_api.txt")
        print("  sudo apt-get install -y poppler-utils")
    print("="*80)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
