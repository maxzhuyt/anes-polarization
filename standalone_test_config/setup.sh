#!/bin/bash
# Setup script for standalone test_config on bare metal server
# Requires: Python 3.9+, CUDA 12.x, ~25GB disk for deps + model

set -e

echo "=== Setting up standalone test_config ==="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.2 driver)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers accelerate
pip install pandas numpy scipy scikit-learn joblib

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate: source venv/bin/activate"
echo ""
echo "Run examples:"
echo "  # Llama 3.1 8B (needs HuggingFace login: huggingface-cli login)"
echo "  python test_config.py --model-id meta-llama/Llama-3.1-8B-Instruct --model-name Llama-3.1-8B-Instruct --demo-fields core --prompt-fmt B"
echo ""
echo "  # Gemma 2 9B (may need HF login for gated model)"
echo "  python test_config.py --model-id google/gemma-2-9b-it --model-name gemma-2-9b-it --demo-fields core --prompt-fmt B"
echo ""
echo "WARNING: RTX 3090 has 24GB VRAM. Use batch-size 8 (default)."
echo "         Gemma 2 9B (~18GB) is tight. If OOM, try --batch-size 4."
