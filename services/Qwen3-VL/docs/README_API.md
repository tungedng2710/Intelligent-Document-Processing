# Qwen3-VL Document Parser API

API dịch vụ phân tích và trích xuất nội dung từ tài liệu PDF và hình ảnh sử dụng mô hình Qwen3-VL với vLLM backend.

## Tính năng

- ✅ Hỗ trợ đầu vào: PDF, PNG, JPG, JPEG
- ✅ Trích xuất văn bản với đúng thứ tự đọc (reading order)
- ✅ Bảng biểu hiện dưới dạng HTML
- ✅ Xử lý PDF nhiều trang
- ✅ Inference nhanh với vLLM
- ✅ API RESTful với FastAPI

## Yêu cầu hệ thống

- Python 3.8+
- CUDA-capable GPU (khuyến nghị)
- poppler-utils (cho xử lý PDF)

## Cài đặt

### 1. Cài đặt dependencies

```bash
# Cài đặt vLLM (đã cài đặt)
pip install vllm

# Cài đặt các thư viện API
pip install -r requirements_api.txt

# Cài đặt poppler-utils cho PDF support
sudo apt-get update && sudo apt-get install -y poppler-utils
```

### 2. Tải mô hình (nếu chưa có)

Mô hình sẽ được tự động tải về khi khởi động API lần đầu. Mặc định sử dụng `Qwen/Qwen2-VL-7B-Instruct`.

Nếu bạn đã fine-tune mô hình, có thể chỉ định đường dẫn:

```bash
export MODEL_PATH=/path/to/your/fine-tuned/model
```

## Sử dụng

### Khởi động API

**Cách 1: Sử dụng script**

```bash
./start_api.sh
```

**Cách 2: Chỉ định tham số**

```bash
python api_vllm.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

**Cách 3: Sử dụng biến môi trường**

```bash
export MODEL_PATH=Qwen/Qwen2-VL-7B-Instruct
export HOST=0.0.0.0
export PORT=8000
export TENSOR_PARALLEL_SIZE=1
./start_api.sh
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "running",
  "model": "Qwen/Qwen2-VL-7B-Instruct",
  "message": "Qwen3-VL Document Parser API is running"
}
```

#### 2. Parse Document

**Endpoint:** `POST /parse`

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PDF hoặc image)

**Response thành công:**
```json
{
  "status": "success",
  "markdown": "## Header\n\nContent...\n\n<table>...</table>"
}
```

**Response lỗi:**
```json
{
  "status": "error",
  "message": "Error description"
}
```

### Ví dụ sử dụng

**1. Với cURL - Image:**

```bash
curl -X POST http://localhost:8000/parse \
    -F "file=@document.jpg" \
    | jq '.'
```

**2. Với cURL - PDF:**

```bash
curl -X POST http://localhost:8000/parse \
    -F "file=@document.pdf" \
    | jq '.'
```

**3. Với Python:**

```python
import requests

# Parse image
with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/parse",
        files={"file": f}
    )
    result = response.json()
    print(result["markdown"])

# Parse PDF
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/parse",
        files={"file": f}
    )
    result = response.json()
    print(result["markdown"])
```

**4. Test script:**

```bash
# Tạo hoặc copy test image/pdf, sau đó chạy:
./test_api.sh
```

## Cấu trúc output

Output markdown tuân theo các quy tắc:

1. **Reading Order**: Đúng thứ tự đọc từ trên xuống, trái sang phải
2. **Block Separation**: Các khối logic được phân tách bằng `\n---\n`
3. **Headers**: Labels được format dưới dạng `## Label Name`
4. **Tables**: Dữ liệu dạng bảng được xuất dưới dạng HTML `<table>`
5. **Multi-page PDF**: Mỗi trang được đánh dấu với `## Page N`

### Ví dụ output:

```markdown
## Invoice

Invoice Number: INV-2024-001
Date: 2024-01-04

---

## Bill To

John Doe
123 Main St
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
Tax: $3.50
**Total: $38.50**
```

## Cấu hình nâng cao

### Multi-GPU

Để sử dụng nhiều GPU:

```bash
python api_vllm.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --tensor-parallel-size 2
```

### Custom Model

Nếu đã fine-tune model:

```bash
python api_vllm.py \
    --model-path ./outputs \
    --port 8000
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application
COPY api_vllm.py .

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "api_vllm.py", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### 1. PDF không xử lý được
- Kiểm tra poppler-utils đã cài đặt: `pdftoppm -v`
- Cài đặt: `sudo apt-get install -y poppler-utils`

### 2. Out of memory
- Giảm `max_model_len` trong `init_model()`
- Giảm `gpu_memory_utilization` (mặc định 0.9)
- Sử dụng model nhỏ hơn hoặc quantized

### 3. Model tải chậm
- Model lớn sẽ mất thời gian tải lần đầu
- Xem xét download trước và sử dụng local path

### 4. API không response
- Kiểm tra logs để xem lỗi
- Kiểm tra port đã được sử dụng chưa: `lsof -i :8000`

## Performance

- **Single GPU (A100 40GB)**: ~2-5s/page
- **Multi-GPU**: Scale tuyến tính với tensor parallel
- **vLLM**: Tối ưu throughput và latency so với vanilla inference

## License

Tuân theo license của Qwen3-VL model.
