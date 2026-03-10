# Qwen3-VL Document Parser

A comprehensive document parsing and analysis system built on Qwen3-VL vision-language model, fine-tuned specifically for Bill of Lading and document extraction tasks.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [API Server](#api-server)
  - [Training](#training)
  - [Inference](#inference)
- [Documentation](#documentation)
- [Requirements](#requirements)

## ✨ Features

- 🎯 **Specialized Document Parsing**: Fine-tuned for Bill of Lading and structured document extraction
- 📊 **Table Detection & Extraction**: Advanced table recognition with HTML output
- 🔄 **Multi-format Support**: PDF, PNG, JPG, JPEG inputs
- ⚡ **Fast Inference**: vLLM backend for optimized performance
- 🎓 **GRPO Training**: Group Relative Policy Optimization for improved accuracy
- 📈 **Reward Functions**: Sophisticated evaluation metrics (TEDS, content similarity, format compliance)
- 🌐 **REST API**: FastAPI-based service for easy integration
- 🖥️ **Gradio Interface**: Interactive web UI for testing
- 📝 **Prompt Library**: Centralized prompt management system

## 📁 Project Structure

```
Qwen3-VL/
├── src/
│   ├── api/                    # API server and client code
│   │   ├── api_vllm.py        # Main API with LoRA support
│   │   ├── api_vllm_2.py      # API for merged models
│   │   ├── app.py             # Gradio web interface
│   │   ├── test_client.py     # API test client
│   │   └── batch_inference_client.py
│   ├── training/              # Model training scripts
│   │   ├── unsloth_finetuning_grpo.py       # GRPO training
│   │   ├── unsloth_finetuning_sft_tagged.py # SFT training
│   │   ├── unsloth_finetuning.py            # Base training
│   │   └── grpo_training_guide.py           # Training documentation
│   ├── inference/             # Inference scripts
│   │   ├── infer.py           # Basic inference
│   │   └── infer_sft_tagged.py # SFT model inference
│   ├── evaluation/            # Evaluation and metrics
│   │   ├── reward_functions.py # Reward calculation
│   │   ├── teds.py            # TEDS metric implementation
│   │   └── test_sample_rewards.py
│   └── utils/                 # Utility functions
│       ├── custom_logits_processor.py
│       ├── unsloth_convert_models.py
│       ├── prompt_library.py  # Prompt management system
│       └── verify_installation.py
├── prompts/                   # Prompt library
│   ├── document_extraction_v1.txt  # API prompts
│   ├── document_extraction_v2_semantic.txt
│   ├── training_base.txt      # Training prompts
│   ├── training_sft_tagged.txt
│   └── README.md             # Prompt documentation
├── scripts/                   # Shell scripts
│   ├── start_api.sh          # Start API with LoRA
│   ├── start_api_2.sh        # Start API with merged model
│   ├── start_vllm_model.sh   # Start vLLM server
│   ├── test_api.sh           # Test API endpoints
│   ├── manage_prompts.py     # Prompt management CLI
│   ├── install.sh            # Installation script
│   └── quickstart.sh         # Quick setup
├── docs/                      # Documentation
│   ├── README_API.md         # API documentation
│   ├── USAGE_GUIDE.md        # Usage guide
│   ├── INDEX.md              # Documentation index
│   ├── SUMMARY.md            # Project summary
│   ├── sample.md             # Sample outputs
│   └── result.md             # Results and metrics
├── models/                    # Model artifacts
│   ├── checkpoints/          # Training checkpoints
│   │   ├── base-sft/        # Base SFT checkpoints
│   │   ├── grpo/            # GRPO training checkpoints
│   │   └── sft-tagged/      # SFT tagged checkpoints
│   ├── lora/                 # LoRA adapters
│   └── merged/               # Merged models
│       └── qwen3_vl_grpo_170/
├── submission/               # Final submission version
│   └── (preserved - do not modify)
├── test_samples/            # Test images and PDFs
├── requirements_api.txt     # API dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── README.md               # This file

```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for inference
- 32GB+ GPU memory for training

### Quick Install

```bash
# Clone repository
cd /path/to/Qwen3-VL

# Install dependencies
./scripts/install.sh

# Or install manually
pip install -r requirements_api.txt
pip install unsloth vllm transformers
```

### System Dependencies

```bash
# For PDF support
sudo apt-get update
sudo apt-get install -y poppler-utils
```

## 🎯 Quick Start

### 1. Start API Server (Fastest)

```bash
# Option A: Use merged model (fastest startup)
export MODEL_PATH=/path/to/merged/model
./scripts/start_api_2.sh

# Option B: Use base model + LoRA
export MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export LORA_PATH=./outputs_sft_tagged/checkpoint-300
./scripts/start_api.sh
```

### 2. Test API

```bash
# Test with curl
curl -X POST http://localhost:9890/parse \
  -F "file=@test_image.jpg"

# Or use test script
./scripts/test_api.sh
```

### 3. Use Web Interface

```bash
# Start Gradio app
cd src/api && python app.py
# Open http://localhost:7860
```

## 📖 Usage

### API Server

#### Start Server

```bash
# Default settings
./scripts/start_api_2.sh

# Custom settings
MODEL_PATH=/custom/path \
HOST=0.0.0.0 \
PORT=8080 \
TENSOR_PARALLEL_SIZE=2 \
./scripts/start_api_2.sh
```

#### API Endpoints

**POST /parse** - Parse document
```bash
curl -X POST http://localhost:9890/parse \
  -F "file=@document.pdf" \
  -F "max_tokens=4096" \
  -F "temperature=0.7"
```

**GET /** - Health check
```bash
curl http://localhost:9890/
```

See [docs/README_API.md](docs/README_API.md) for detailed API documentation.

### Training

#### GRPO Training (Recommended)

```bash
cd src/training
python unsloth_finetuning_grpo.py
```

Features:
- Group Relative Policy Optimization
- Multiple reward functions
- TensorBoard logging
- Automatic checkpointing

#### SFT Training

```bash
cd src/training
python unsloth_finetuning_sft_tagged.py
```

See [src/training/grpo_training_guide.py](src/training/grpo_training_guide.py) for training details.

### Inference

#### Basic Inference

```bash
cd src/inference
python infer.py --image /path/to/image.jpg
```

#### Batch Inference

```bash
cd src/api
python batch_inference_client.py \
  --input-dir /path/to/images \
  --output-dir /path/to/results
```

## 📚 Documentation

- [API Documentation](docs/README_API.md) - Complete API reference
- [Usage Guide](docs/USAGE_GUIDE.md) - Detailed usage instructions
- [Training Guide](src/training/grpo_training_guide.py) - Training methodology
- [Sample Outputs](docs/sample.md) - Example extractions
- [Results & Metrics](docs/result.md) - Performance evaluation

## 🔧 Configuration

### Environment Variables

```bash
# Model paths
MODEL_PATH=/path/to/model
LORA_PATH=/path/to/lora

# Server config
HOST=0.0.0.0
PORT=9890
TENSOR_PARALLEL_SIZE=1

# Training config
OUTPUT_DIR=./outputs
MAX_STEPS=1000
LEARNING_RATE=5e-5
```

### Model Selection

1. **Base Model + LoRA** (Flexible)
   - Path: `Qwen/Qwen3-VL-8B-Instruct` + LoRA checkpoint
   - Use: `scripts/start_api.sh`

2. **Merged Model** (Faster)
   - Path: Pre-merged model directory
   - Use: `scripts/start_api_2.sh`

3. **Custom Fine-tuned**
   - Path: Your training output
   - Use: Either script with custom MODEL_PATH

### Prompt Management

The system uses a centralized prompt library for easy management and versioning.

#### List Available Prompts

```bash
python scripts/manage_prompts.py list
```

#### View a Prompt

```bash
# Using alias
python scripts/manage_prompts.py show api

# Using full name
python scripts/manage_prompts.py show document_extraction_v1
```

#### Create New Prompt

```bash
# From content
python scripts/manage_prompts.py create my_prompt --content "**Task:** ..."

# From file
python scripts/manage_prompts.py create my_prompt --file prompt.txt

# Interactive
python scripts/manage_prompts.py create my_prompt
```

#### Using Prompts in Code

```python
from utils.prompt_library import get_prompt

# Load prompt by alias
prompt = get_prompt('api')

# Or by full name
prompt = get_prompt('document_extraction_v1')
```

See [prompts/README.md](prompts/README.md) for complete documentation.

## 📊 Evaluation Metrics

The system uses multiple reward functions for evaluation:

- **Block Classification**: Correct text_block vs table_block classification
- **Table Accuracy**: TEDS-based table structure similarity
- **Content Similarity**: Semantic text matching
- **Format Compliance**: Markdown/HTML format correctness
- **Reading Order**: Correct sequence of extracted content

See [src/evaluation/reward_functions.py](src/evaluation/reward_functions.py) for implementation.

## 🐳 Docker Support

```bash
# Build image
docker-compose build

# Run service
docker-compose up -d

# View logs
docker-compose logs -f
```

## 🧪 Testing

```bash
# Test reward functions
cd src/evaluation
python test_sample_rewards.py

# Test API
./scripts/test_api.sh

# Verify installation
cd src/utils
python verify_installation.py
```

## 📝 Requirements

### Core Dependencies
- `torch>=2.0.0`
- `transformers>=4.40.0`
- `vllm>=0.3.0`
- `unsloth>=2024.1`
- `fastapi>=0.109.0`
- `gradio>=4.0.0`
- `pdf2image>=1.16.0`
- `pillow>=10.0.0`

See [requirements_api.txt](requirements_api.txt) for complete list.

## 🤝 Contributing

The submission folder contains the final version and should not be modified. Development work should be done in the main src directory.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Qwen Team for the base model
- Unsloth for optimization tools
- vLLM for efficient inference

## 📮 Contact

For questions or issues, please [create an issue](https://github.com/your-repo/issues) or contact the maintainers.

---

**Note**: The `submission/` directory contains the final working version and should be kept unchanged.
