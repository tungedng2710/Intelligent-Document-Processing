# Qwen3-VL Document Parser API

API dịch vụ phân tích và trích xuất nội dung từ tài liệu PDF và hình ảnh sử dụng mô hình Qwen3-VL với vLLM backend.

## 🚀 Quick Start

```bash
# 1. Run verification
python verify_installation.py

# 2. Start API
./start_api.sh

# 3. Test API
curl http://localhost:8000/
```

hoặc sử dụng interactive guide:

```bash
./quickstart.sh
```

## 📁 Project Structure

```
Qwen3-VL/
├── api_vllm.py              # Main API server (FastAPI + vLLM)
├── requirements_api.txt     # Python dependencies
├── start_api.sh            # Script khởi động API
├── quickstart.sh           # Interactive quick start guide
├── verify_installation.py  # Kiểm tra dependencies
│
├── test_api.sh             # Test với curl
├── test_client.py          # Python test client
│
├── SUMMARY.md              # ⭐ Tổng quan và quick start
├── README_API.md           # 📖 API documentation chi tiết
├── USAGE_GUIDE.md          # 📚 Hướng dẫn sử dụng đầy đủ
├── INDEX.md                # 📋 File này - navigation guide
│
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
│
├── infer.py               # Original inference script (unsloth)
└── outputs/               # Fine-tuned model checkpoints
```

## 📚 Documentation Guide

### Bắt đầu nhanh
👉 **[SUMMARY.md](SUMMARY.md)** - Đọc đầu tiên để có cái nhìn tổng quan
- Tính năng đã implement
- Quick start commands
- System requirements
- Troubleshooting cơ bản

### API Documentation
👉 **[README_API.md](README_API.md)** - Chi tiết về API
- API endpoints và parameters
- Configuration options
- Performance tuning
- Production deployment
- Multi-GPU setup

### Hướng dẫn sử dụng
👉 **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Examples và tutorials
- Cách sử dụng với curl, Python, JavaScript
- Ví dụ output format
- Common use cases
- Best practices
- FAQ

## 🎯 API Specification

### Endpoint: `POST /parse`

**Input:** PDF hoặc image file (png, jpg, jpeg)

**Output:** JSON
```json
{
  "status": "success|error",
  "markdown": "...",  // khi success
  "message": "..."    // khi error
}
```

**Features:**
- ✅ Giữ đúng reading order
- ✅ Bảng dạng HTML
- ✅ Hỗ trợ PDF nhiều trang
- ✅ Fast inference với vLLM

## 🔧 Main Files Explained

### Core Files

**[api_vllm.py](api_vllm.py)**
- Main API server
- FastAPI application
- vLLM model loading
- Request/response handling
- PDF processing logic

**[start_api.sh](start_api.sh)**
- Convenient startup script
- Environment variable support
- Automatic dependency check
- Production-ready settings

### Testing Tools

**[test_client.py](test_client.py)**
- Python client library
- CLI tool for testing
- Usage examples

```bash
# Health check
python test_client.py health

# Parse document
python test_client.py parse --file doc.pdf --output result.md
```

**[test_api.sh](test_api.sh)**
- Bash script for testing
- curl-based tests
- Quick validation

**[verify_installation.py](verify_installation.py)**
- Check all dependencies
- System requirements validation
- GPU/CUDA check

### Configuration Files

**[requirements_api.txt](requirements_api.txt)**
- Python package dependencies
- Version specifications

**[Dockerfile](Dockerfile)**
- Container image definition
- For Docker deployment

**[docker-compose.yml](docker-compose.yml)**
- Multi-container setup
- GPU configuration
- Volume mounts

## 💻 Usage Examples

### 1. Start API

```bash
# Simple start
./start_api.sh

# Custom model
python api_vllm.py --model-path outputs --port 8000

# Multi-GPU
python api_vllm.py --tensor-parallel-size 2
```

### 2. Test API

**With curl:**
```bash
curl -X POST http://localhost:8000/parse \
    -F "file=@document.pdf" | jq '.'
```

**With Python:**
```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/parse",
        files={"file": f}
    )
    result = response.json()
    print(result["markdown"])
```

**With test client:**
```bash
python test_client.py parse --file document.pdf --output result.md
```

### 3. Docker Deployment

```bash
# Build image
docker build -t qwen3vl-api .

# Run container
docker run --gpus all -p 8000:8000 qwen3vl-api

# Or use docker-compose
docker-compose up -d
```

## 🛠️ Configuration

### Environment Variables

```bash
export MODEL_PATH=Qwen/Qwen2-VL-7B-Instruct
export HOST=0.0.0.0
export PORT=8000
export TENSOR_PARALLEL_SIZE=1
```

### Command Line Arguments

```bash
python api_vllm.py \
    --model-path <path_to_model> \
    --host <host> \
    --port <port> \
    --tensor-parallel-size <num_gpus>
```

## 📊 System Status

### Verified Components
✅ Python 3.12.12  
✅ 2x NVIDIA A30 GPU  
✅ CUDA available  
✅ vLLM installed  
✅ FastAPI installed  
✅ pdf2image installed  
✅ poppler-utils installed  

## 🔍 Common Tasks

### Check API health
```bash
curl http://localhost:8000/
```

### Parse a document
```bash
curl -X POST http://localhost:8000/parse -F "file=@doc.pdf"
```

### View logs
```bash
# If running with start_api.sh, logs show in terminal
# Or redirect to file:
python api_vllm.py 2>&1 | tee api.log
```

### Stop server
```bash
# If running in foreground: Ctrl+C
# If running as service: sudo systemctl stop qwen3vl-api
```

## 🐛 Troubleshooting

### API won't start
1. Run: `python verify_installation.py`
2. Check logs for errors
3. Ensure GPU available: `nvidia-smi`

### Out of memory
- Reduce `gpu_memory_utilization` in api_vllm.py
- Use smaller `max_model_len`
- Try on machine with more GPU RAM

### PDF processing fails
- Check poppler: `pdftoppm -v`
- Install if needed: `sudo apt-get install poppler-utils`

### Slow inference
- Check GPU usage: `nvidia-smi`
- Use tensor parallelism for multi-GPU
- Consider using quantized model

## 📖 Additional Resources

### Model Information
- Base model: Qwen/Qwen2-VL-7B-Instruct
- Vision-Language model for document understanding
- Supports images up to high resolution

### Related Files
- `infer.py` - Original inference script với unsloth
- `outputs/` - Fine-tuned model checkpoints (nếu có)

### External Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qwen Model Card](https://huggingface.co/Qwen)

## 🎓 Learning Path

**Beginner:**
1. Read [SUMMARY.md](SUMMARY.md)
2. Run `./quickstart.sh`
3. Test with sample files using `test_client.py`

**Intermediate:**
4. Read [README_API.md](README_API.md)
5. Customize model and parameters
6. Integrate into your application

**Advanced:**
7. Read [USAGE_GUIDE.md](USAGE_GUIDE.md)
8. Setup production deployment
9. Optimize for your use case
10. Fine-tune model for your documents

## 🤝 Support

For issues:
1. Check [USAGE_GUIDE.md](USAGE_GUIDE.md) FAQ section
2. Run `python verify_installation.py`
3. Check logs for error messages
4. Review troubleshooting sections in documentation

## 📄 License

Follows Qwen model license terms.

---

**Quick Links:**
- 🚀 [SUMMARY.md](SUMMARY.md) - Start here
- 📖 [README_API.md](README_API.md) - API details
- 📚 [USAGE_GUIDE.md](USAGE_GUIDE.md) - Full guide
- 🔧 [api_vllm.py](api_vllm.py) - Source code

**Ready to start?** Run `./quickstart.sh` or `./start_api.sh`
