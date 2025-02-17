# 🚀 Local LLM Chat with Hugging Face Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)

This repository accompanies the [YouTube tutorial](https://www.youtube.com/@BahrainAI) demonstrating how to run a local large language model (LLM) using Hugging Face Transformers. The provided Python script implements a complete workflow for chat-style interactions with a locally stored model.

[LLM MODEL](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

## 📚 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Code Structure](#-code-structure)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)
- [Resources](#-resources)

## 🌟 Features
- **Local Model Loading** 🔄  
  Load pretrained causal language models from local checkpoints
- **Chat Template Formatting** 💬  
  Supports conversational message formatting
- **Smart Tokenization** 🔠  
  Includes attention masking and device allocation
- **Controlled Generation** 🎛️  
  Configurable parameters:
  - Temperature
  - Top-p sampling
  - Max new tokens
- **Cross-Platform Support** 💻  
  Works on CPU (CUDA supported for GPU acceleration)

## 🛠️ Prerequisites
- Python 3.7+
- [PyTorch](https://pytorch.org) (CPU/CUDA version)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- 8GB+ RAM (16GB recommended)

## 📥 Installation
```bash
# Clone repository
git clone https://github.com/portalbh/SmolLM2.git
cd SmolLM2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows

# Install dependencies
pip install torch transformers
