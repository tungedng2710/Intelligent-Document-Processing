# Hướng dẫn sử dụng Qwen3-VL Document Parser API

## Tóm tắt

API này nhận đầu vào là file PDF hoặc ảnh và trả về nội dung được trích xuất dưới dạng markdown, với các bảng được biểu diễn bằng HTML.

## Khởi động API

### Bước 1: Cài đặt dependencies

```bash
cd /root/tungn197/license_plate_recognition/services/Qwen3-VL
pip install -r requirements_api.txt
```

### Bước 2: Cài đặt poppler-utils (cho PDF)

```bash
sudo apt-get update && sudo apt-get install -y poppler-utils
```

### Bước 3: Khởi động server

**Cách đơn giản nhất:**
```bash
./start_api.sh
```

**Hoặc với custom model:**
```bash
python api_vllm.py --model-path /path/to/your/model --port 8000
```

Server sẽ chạy tại `http://localhost:8000`

## Sử dụng API

### 1. Kiểm tra API đang chạy

```bash
curl http://localhost:8000/
```

Kết quả:
```json
{
  "status": "running",
  "model": "Qwen/Qwen2-VL-7B-Instruct",
  "message": "Qwen3-VL Document Parser API is running"
}
```

### 2. Parse một file ảnh

```bash
curl -X POST http://localhost:8000/parse \
    -F "file=@/path/to/document.jpg" \
    -o result.json
```

### 3. Parse một file PDF

```bash
curl -X POST http://localhost:8000/parse \
    -F "file=@/path/to/document.pdf" \
    -o result.json
```

### 4. Sử dụng Python client

```bash
# Kiểm tra health
python test_client.py health

# Parse document
python test_client.py parse --file document.jpg

# Parse và lưu kết quả
python test_client.py parse --file document.pdf --output result.md

# Với pretty print
python test_client.py parse --file document.jpg --pretty
```

### 5. Sử dụng trong code Python

```python
import requests

def parse_document(file_path, api_url="http://localhost:8000"):
    """Parse a document using the API."""
    with open(file_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/parse",
            files={'file': f}
        )
    
    result = response.json()
    
    if result['status'] == 'success':
        return result['markdown']
    else:
        raise Exception(result['message'])

# Sử dụng
markdown = parse_document('invoice.pdf')
print(markdown)
```

### 6. Sử dụng trong JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function parseDocument(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    const response = await axios.post(
        'http://localhost:8000/parse',
        form,
        { headers: form.getHeaders() }
    );
    
    if (response.data.status === 'success') {
        return response.data.markdown;
    } else {
        throw new Error(response.data.message);
    }
}

// Sử dụng
parseDocument('document.pdf')
    .then(markdown => console.log(markdown))
    .catch(error => console.error(error));
```

## Cấu trúc Response

### Success Response

```json
{
  "status": "success",
  "markdown": "## Header\n\nContent here...\n\n<table>\n<tr><th>Col1</th><th>Col2</th></tr>\n<tr><td>Data1</td><td>Data2</td></tr>\n</table>"
}
```

### Error Response

```json
{
  "status": "error",
  "message": "Invalid file type. Allowed: .pdf, .png, .jpg, .jpeg"
}
```

## Ví dụ Output Markdown

### Ví dụ 1: Invoice đơn giản

```markdown
## Invoice

Invoice Number: INV-2024-001
Date: January 4, 2024

---

## Bill To

ABC Company
123 Business St
City, State 12345

---

## Items

<table>
<tr><th>Description</th><th>Quantity</th><th>Unit Price</th><th>Amount</th></tr>
<tr><td>Consulting Services</td><td>10 hours</td><td>$150.00</td><td>$1,500.00</td></tr>
<tr><td>Software License</td><td>1</td><td>$500.00</td><td>$500.00</td></tr>
</table>

---

## Total

Subtotal: $2,000.00
Tax (10%): $200.00
**Total Due: $2,200.00**
```

### Ví dụ 2: PDF nhiều trang

```markdown
## Page 1

## Contract Agreement

This agreement is made on January 4, 2024...

---

## Parties

**Party A:** John Doe
**Party B:** Jane Smith

---

## Page 2

## Terms and Conditions

1. Payment terms...
2. Delivery schedule...
3. Warranty...

---

<table>
<tr><th>Milestone</th><th>Date</th><th>Payment</th></tr>
<tr><td>Start</td><td>Jan 15</td><td>$10,000</td></tr>
<tr><td>Midpoint</td><td>Feb 15</td><td>$10,000</td></tr>
<tr><td>Completion</td><td>Mar 15</td><td>$10,000</td></tr>
</table>
```

## Xử lý Errors

### Common Errors và Solutions

1. **"Model not initialized"**
   - Solution: Đợi model load xong (có thể mất vài phút lần đầu)

2. **"Invalid file type"**
   - Solution: Chỉ upload file .pdf, .png, .jpg, .jpeg

3. **"Error converting PDF to images"**
   - Solution: Cài đặt poppler-utils: `sudo apt-get install poppler-utils`

4. **"Connection refused"**
   - Solution: Kiểm tra API đang chạy: `curl http://localhost:8000/`

5. **Request timeout**
   - Solution: File quá lớn hoặc phức tạp. Thử giảm kích thước file hoặc tăng timeout

## Performance Tips

1. **Tối ưu cho throughput cao:**
   - Sử dụng batch processing
   - Scale horizontal với nhiều instance API

2. **Tối ưu cho latency thấp:**
   - Sử dụng GPU mạnh (A100, V100)
   - Tăng tensor_parallel_size cho multi-GPU

3. **Tối ưu cho file lớn:**
   - Tăng max_model_len nếu cần
   - Sử dụng ảnh resolution thấp hơn nếu có thể

## Monitoring

### Logs

API sẽ ghi logs ra console. Để lưu logs:

```bash
python api_vllm.py 2>&1 | tee api.log
```

### Health Check Script

Tạo script để monitor:

```bash
#!/bin/bash
while true; do
    if curl -s http://localhost:8000/ > /dev/null; then
        echo "$(date): API is healthy"
    else
        echo "$(date): API is down!"
        # Restart hoặc alert
    fi
    sleep 60
done
```

## Production Deployment

### Sử dụng systemd

Tạo file `/etc/systemd/system/qwen3vl-api.service`:

```ini
[Unit]
Description=Qwen3-VL Document Parser API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/root/tungn197/license_plate_recognition/services/Qwen3-VL
ExecStart=/usr/bin/python3 api_vllm.py --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable và start:
```bash
sudo systemctl enable qwen3vl-api
sudo systemctl start qwen3vl-api
sudo systemctl status qwen3vl-api
```

### Sử dụng với Nginx reverse proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

## Câu hỏi thường gặp

**Q: API có hỗ trợ batch processing không?**
A: Hiện tại mỗi request xử lý 1 file. Để batch, gọi API nhiều lần song song.

**Q: Có giới hạn kích thước file không?**
A: Phụ thuộc vào GPU memory. Khuyến nghị < 20MB cho image, < 50 trang cho PDF.

**Q: Có thể thay đổi prompt instruction không?**
A: Có, chỉnh sửa hàm `get_instruction_prompt()` trong `api_vllm.py`.

**Q: Có hỗ trợ OCR không?**
A: Model Qwen3-VL tự động OCR, không cần external OCR engine.

**Q: Độ chính xác như thế nào?**
A: Phụ thuộc vào chất lượng ảnh và model. Fine-tuned model cho kết quả tốt hơn.
