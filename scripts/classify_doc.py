import base64
import json
import requests
import sys
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

# Configuration
OLLAMA_BASE_URL = "http://localhost:7860"
MODEL_NAME = "qwen3.5:9b-bf16"

# Document type mappings from doctype.md
DOCUMENT_TYPES = {
    "PHIẾU KHÁM BỆNH VÀO VIỆN": {"json": "page-03.json", "template": "page-03-template.json"},
    "PHIẾU CHỈ ĐỊNH CẬN LÂM SÀNG": {"json": "page-04.json", "template": "page-04-template.json"},
    "PHIẾU TRẢ KẾT QUẢ HUYẾT HỌC": {"json": "page-23.json", "template": "page-23-template.json"},
    "PHIẾU CHĂM SÓC": {"json": "page-24.json", "template": "page-24-template.json"},
    "PHIẾU THEO DÕI CHỨC NĂNG SỐNG": {"json": "page-36.json", "template": "page-36-template.json"},
    "TỜ ĐIỀU TRỊ": {"json": "page-38.json", "template": "page-38-template.json"},
    "GIẤY RA VIỆN": {"json": "page-60.json", "template": "page-60-template.json"},
    "BẢNG KÊ CHI PHÍ ĐIỀU TRỊ NGOẠI TRÚ": {"json": "page-61.json", "template": "page-61-template.json"},
    "PHIẾU CÔNG KHAI DỊCH VỤ KCB NỘI TRÚ": {"json": "page-62.json", "template": "page-62-template.json"},
}

DOCUMENT_NAMES = list(DOCUMENT_TYPES.keys())


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def classify_document(image_path: str, base_url: str = OLLAMA_BASE_URL, model: str = MODEL_NAME) -> Tuple[Optional[str], Optional[str]]:
    """
    Classify a document image using Ollama LLM.
    
    Args:
        image_path: Path to the image file
        base_url: Ollama base URL (default: http://localhost:7860)
        model: Model name (default: qwen3.5:9b-bf16)
    
    Returns:
        Tuple of (document_type, template_filename) or (None, None) if classification fails
    """
    try:
        # Validate image exists
        if not Path(image_path).exists():
            print(f"Error: Image file not found: {image_path}")
            return None, None
        
        # Encode image to base64
        image_b64 = encode_image_to_base64(image_path)
        
        # Prepare the prompt
        document_list = "\n".join([f"- {doc}" for doc in DOCUMENT_NAMES])
        prompt = f"""You are a healthcare document classifier. Analyze this medical document image and identify its type.

Available document types:
{document_list}

Please respond with ONLY the exact document type name from the list above. Do not include any explanation or additional text."""
        
        # Call Ollama API
        url = f"{base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        
        print(f"Calling Ollama at {url} with model {model}...")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        classification_text = result.get("response", "").strip()
        
        print(f"LLM Response: {classification_text}")
        
        # Find matching document type
        for doc_type in DOCUMENT_NAMES:
            if doc_type.lower() in classification_text.lower():
                template_info = DOCUMENT_TYPES[doc_type]
                return doc_type, template_info["template"]
        
        print(f"Warning: Could not match response to any document type")
        return None, None
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {base_url}")
        print(f"Please ensure Ollama is running on port 7860")
        return None, None
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return None, None


def main():
    """Main function for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python classify_doc.py <image_path> [ollama_base_url] [model_name]")
        print(f"\nDefault Ollama URL: {OLLAMA_BASE_URL}")
        print(f"Default Model: {MODEL_NAME}")
        print("\nExample:")
        print("  python classify_doc.py /path/to/document.png")
        print("  python classify_doc.py /path/to/document.png http://localhost:7860 qwen3.5:9b-bf16")
        sys.exit(1)
    
    image_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else OLLAMA_BASE_URL
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL_NAME
    
    print(f"Classifying document: {image_path}")
    doc_type, template = classify_document(image_path, base_url, model)
    
    if doc_type and template:
        print(f"\nClassification Result:")
        print(f"  Document Type: {doc_type}")
        print(f"  Template File: {template}")
        output = {
            "document_type": doc_type,
            "template_filename": template,
            "status": "success"
        }
    else:
        print(f"\nClassification failed")
        output = {
            "document_type": None,
            "template_filename": None,
            "status": "failed"
        }
    
    print(f"\nJSON Output:")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    main()
