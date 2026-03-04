# Document Analysis Web Application

A modern and professional web application for document analysis with markdown preview and DOCX conversion capabilities.

## Features

- 📤 **File Upload**: Drag-and-drop or browse to upload documents (PDF, PNG, JPG, etc.)
- 🔍 **Document Analysis**: Automatically sends files to analysis service and retrieves markdown
- 👁️ **Markdown Preview**: Beautiful real-time preview of analyzed documents
- 📥 **DOCX Export**: Convert markdown results to downloadable DOCX files
- ⚙️ **Configurable Service**: Update analysis service URL on the fly
- 🎨 **Modern UI**: Professional gradient design with responsive layout

## Architecture

### Backend (FastAPI)
- **POST /api/analyze**: Upload file and analyze via external service
- **POST /api/convert-to-docx**: Convert markdown to DOCX format
- **GET /api/health**: Check service status
- **POST /api/update-service-url**: Update analysis service endpoint

### Frontend
- Single-page application with vanilla JavaScript
- Drag-and-drop file upload
- Real-time markdown rendering
- DOCX download functionality

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

The application will start on `http://0.0.0.0:8000`

## Configuration

### Analysis Service URL
Default: `http://0.0.0.0:9890/parse`

You can change this in three ways:

1. **Environment Variable**:
```bash
export ANALYSIS_SERVICE_URL="http://your-service:port/parse"
python main.py
```

2. **UI Settings**: Use the configuration section at the bottom of the web interface

3. **API Call**:
```bash
curl -X POST http://localhost:8000/api/update-service-url \
  -H "Content-Type: application/json" \
  -d '{"new_url": "http://your-service:port/parse"}'
```

## Usage

1. Open your browser to `http://localhost:8000`
2. Upload a document by:
   - Dragging and dropping onto the upload zone
   - Clicking "Browse Files" to select a file
3. Click "Analyze Document" to process the file
4. View the markdown preview of the results
5. (Optional) Click "Download as DOCX" to export to Word format

## API Examples

### Analyze Document
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@document.pdf"
```

### Convert to DOCX
```bash
curl -X POST http://localhost:8000/api/convert-to-docx \
  -F "markdown_text=# My Document\n\nContent here..." \
  -F "filename=document.pdf" \
  --output document.docx
```

### Health Check
```bash
curl http://localhost:8000/api/health
```

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- HTTPX (for async HTTP requests)
- python-docx (for DOCX generation)
- BeautifulSoup4 (for HTML parsing)
- Markdown library

## Directory Structure

```
app_document_analysis/
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── static/
│   └── index.html         # Frontend UI
└── uploads/               # Temporary file storage
```

## Notes

- Uploaded files are temporarily stored and automatically cleaned up after processing
- The application supports various document formats (PDF, PNG, JPG, JPEG, TIFF, BMP)
- DOCX conversion includes support for headers, paragraphs, lists, tables, and code blocks
- The analysis service must accept multipart/form-data file uploads and return JSON with a "markdown" field

## Troubleshooting

**Analysis Service Offline**:
- Check if the analysis service is running at the configured URL
- Verify the service URL in the configuration section
- Check network connectivity

**File Upload Errors**:
- Ensure the file format is supported
- Check file size limits (default: depends on FastAPI configuration)
- Verify file permissions

**DOCX Conversion Issues**:
- Ensure python-docx is properly installed
- Check that markdown content is valid

## License

MIT License
