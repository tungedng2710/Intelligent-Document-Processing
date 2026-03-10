# Qwen3-VL Document Parser API - Summary

## ✅ Đã hoàn thành

Đã tạo xong API service cho Qwen3-VL với đầy đủ yêu cầu:

### Các file đã tạo:

1. **[api_vllm.py](api_vllm.py)** - Main API server sử dụng vLLM và FastAPI
2. **[requirements_api.txt](requirements_api.txt)** - Dependencies cho API
3. **[start_api.sh](start_api.sh)** - Script khởi động API
4. **[test_api.sh](test_api.sh)** - Script test API với curl
5. **[test_client.py](test_client.py)** - Python client để test API
6. **[verify_installation.py](verify_installation.py)** - Script kiểm tra dependencies
7. **[README_API.md](README_API.md)** - Documentation chi tiết
8. **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Hướng dẫn sử dụng đầy đủ

### Tính năng đã implement:

✅ Endpoint `/parse` với method POST  
✅ Nhận đầu vào: PDF, PNG, JPG, JPEG  
✅ Trả về JSON với format yêu cầu:
- Success: `{"status": "success", "markdown": "..."}`
- Error: `{"status": "error", "message": "..."}`  
✅ Markdown giữ đúng reading order  
✅ Bảng được biểu diễn dưới dạng HTML  
✅ Hỗ trợ PDF nhiều trang  
✅ Sử dụng vLLM cho inference nhanh  
✅ Multi-GPU support với tensor parallelism  

## 🚀 Cách sử dụng nhanh

### 1. Khởi động API

```bash
cd /root/tungn197/license_plate_recognition/services/Qwen3-VL

# Cách 1: Sử dụng script
./start_api.sh

# Cách 2: Chạy trực tiếp với custom model
python api_vllm.py --model-path Qwen/Qwen2-VL-7B-Instruct --port 8000
```

### 2. Test API

```bash
# Health check
curl http://localhost:8000/

# Parse image
curl -X POST http://localhost:8000/parse \
    -F "file=@image.jpg" | jq '.'

# Parse PDF
curl -X POST http://localhost:8000/parse \
    -F "file=@document.pdf" | jq '.'
```

### 3. Sử dụng Python client

```bash
# Health check
python test_client.py health

# Parse document
python test_client.py parse --file document.jpg --output result.md
```

## 📋 API Specification

### Endpoint: POST /parse

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (PDF hoặc image)

**Response Format:**

Success:
```json
{
  "status": "success",
  "markdown": "## Header\n\nContent...\n\n<table>...</table>"
}
```

Error:
```json
{
  "status": "error",
  "message": "Error description"
}
```

### Endpoint: GET /

Health check endpoint.

```json
{
  "status": "running",
  "model": "Qwen/Qwen2-VL-7B-Instruct",
  "message": "Qwen3-VL Document Parser API is running"
}
```

## 🔧 Cấu hình

### Environment Variables

```bash
export MODEL_PATH=Qwen/Qwen2-VL-7B-Instruct  # Đường dẫn model
export HOST=0.0.0.0                           # Host để bind
export PORT=8000                              # Port
export TENSOR_PARALLEL_SIZE=1                 # Số GPU sử dụng
```

### Command Line Arguments

```bash
python api_vllm.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

## 💡 Ví dụ Output

### Input: Invoice image

### Output Markdown:

```markdown
## Invoice

Invoice Number: INV-2024-001
Date: January 4, 2024

---

## Bill To

ABC Company
123 Business Street
City, State 12345

---

## Items

<table>
<tr><th>Item</th><th>Quantity</th><th>Price</th><th>Total</th></tr>
<tr><td>Product A</td><td>2</td><td>$10.00</td><td>$20.00</td></tr>
<tr><td>Product B</td><td>1</td><td>$15.00</td><td>$15.00</td></tr>
</table>

---

## Total

Subtotal: $35.00
Tax (10%): $3.50
**Grand Total: $38.50**
```

## 📊 System Requirements

- ✅ Python 3.8+ (Đã có: 3.12.12)
- ✅ CUDA GPU (Đã có: 2x NVIDIA A30)
- ✅ vLLM (Đã cài)
- ✅ FastAPI + Uvicorn (Đã cài)
- ✅ pdf2image (Đã cài)
- ✅ poppler-utils (Đã cài)

## 🎯 Next Steps

### Để chạy với fine-tuned model:

Nếu bạn đã fine-tune model ở thư mục `outputs/`, chỉ cần:

```bash
python api_vllm.py --model-path outputs --port 8000
```

Hoặc sửa trong `start_api.sh`:

```bash
export MODEL_PATH=outputs
./start_api.sh
```

### Để test với file thật:

```bash
# Chuẩn bị test file
cp /path/to/your/test.jpg .

# Test với Python client
python test_client.py parse --file test.jpg --output result.md --pretty

# Xem kết quả
cat result.md
```

### Để deploy production:

1. **Sử dụng systemd** (xem [USAGE_GUIDE.md](USAGE_GUIDE.md))
2. **Setup Nginx reverse proxy** cho SSL
3. **Monitoring & logging** với systemd journals
4. **Auto-restart** với systemd

## 🐛 Troubleshooting

### API không start?

```bash
# Kiểm tra dependencies
python verify_installation.py

# Xem logs chi tiết
python api_vllm.py --model-path Qwen/Qwen2-VL-7B-Instruct
```

### Out of memory?

Trong [api_vllm.py](api_vllm.py), giảm:
- `gpu_memory_utilization=0.9` → `0.7`
- `max_model_len=8192` → `4096`

### PDF không parse được?

```bash
# Kiểm tra poppler
pdftoppm -v

# Cài nếu chưa có
sudo apt-get install -y poppler-utils
```

## 📚 Documentation

- **[README_API.md](README_API.md)** - Chi tiết về API, cấu hình, performance
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Hướng dẫn sử dụng đầy đủ, examples
- **[api_vllm.py](api_vllm.py)** - Source code với comments đầy đủ

## ✨ Key Features

1. **High Performance**: vLLM backend cho inference nhanh
2. **Multi-GPU Support**: Tensor parallelism cho scale
3. **PDF Support**: Tự động convert PDF thành images
4. **Structured Output**: Markdown với HTML tables
5. **Reading Order**: Giữ nguyên thứ tự đọc tài liệu
6. **Error Handling**: Comprehensive error messages
7. **Easy Testing**: Nhiều tools để test (curl, Python client)
8. **Production Ready**: Systemd, Nginx, monitoring support

## 🎉 Kết luận

API đã sẵn sàng sử dụng! Tất cả requirements đã được implement đầy đủ:

✅ Endpoint `/parse` với POST method  
✅ Nhận PDF/image input  
✅ Trả về JSON với đúng format  
✅ Markdown với reading order và HTML tables  
✅ Full documentation và testing tools  

Bắt đầu bằng:
```bash
./start_api.sh
```

Chúc may mắn! 🚀
