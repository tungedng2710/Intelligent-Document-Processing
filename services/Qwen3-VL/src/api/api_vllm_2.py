from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
import os
import sys
import tempfile
import logging
from pathlib import Path
import uvicorn
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from PIL import Image
import pdf2image
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.prompt_library import get_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Qwen3-VL Document Parser API")

# Global model variable
llm = None
model_name = None

# Default paths - using merged model
DEFAULT_MERGED_MODEL = "services/Qwen3-VL/merged_model_checkpoint-2000"

# Response models
class ParseResponse(BaseModel):
    status: str
    markdown: Optional[str] = None
    message: Optional[str] = None


def init_model(
    model_path: str = DEFAULT_MERGED_MODEL,
    tensor_parallel_size: int = 1
):
    """Initialize the vLLM model (merged model with LoRA weights already integrated)."""
    global llm, model_name
    
    logger.info(f"Initializing vLLM model: {model_path}")
    logger.info("Using merged model (LoRA weights already integrated)")
    
    try:
        # Build LLM kwargs for merged model
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 16384,
            "limit_mm_per_prompt": {"image": 1},
            "dtype": "bfloat16",
            "trust_remote_code": True,  # Required for Qwen3-VL
            "allowed_local_media_path": "/",  # Allow loading local image files
        }
        
        llm = LLM(**llm_kwargs)
        model_name = model_path
        
        logger.info(f"Merged model {model_path} loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


def get_instruction_prompt():
    """Load instruction prompt from prompt library."""
    try:
        return get_prompt('api')
    except Exception as e:
        logger.warning(f"Failed to load prompt from library: {e}. Using fallback.")
        # Fallback prompt
        return """
**Task:** Perform document layout analysis and content extraction. Extract all text from the image and organize it into logical blocks (based on both visual cues, border lines, white space and semantic meanings).
Classify each block whether it is table or not before output its content in markdown or html.

**Format:**
- Wrap each block: `<text_block>content</text_block>` or `<table_block>content</table_block>`
- Headers: If block has a header, start with `##` else leave empty.
- Tables: HTML `<table>` format
- Checkboxes: `[ ]` empty, `[x]` checked
- Special symbols: HTML entities (&copy; &reg; &diam; &cir; &deg; &amp; etc.)
- Transcribe text exactly as shown.

**Output:**
"""


def convert_pdf_to_images(pdf_path: str) -> list:
    """Convert PDF to list of PIL Images."""
    try:
        images = pdf2image.convert_from_path(pdf_path)
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise

import re

def postprocess_markdown(text: str) -> str:
    """Post-process generated markdown text."""
    # Example: Fix common issues, trim whitespace, etc.
    if "<text_block>" in text or "<table_block>" in text:
        # Basic cleanup: ensure proper newlines around blocks
        text = text.replace("<text_block>", "").replace("</text_block>", "\n---")
        text = text.replace("<table_block>", "").replace("</table_block>", "\n---")
    
    # Convert checkbox symbols to markdown format
    checkbox_replacements = {
        '☐': '[ ]',  # Empty box
        '☑': '[x]',  # Checked box
        '☒': '[x]',  # X-marked box
        '✓': '[x]',  # Checkmark
        '✔': '[x]',  # Heavy checkmark
        '✗': '[ ]',  # X mark (treating as unchecked)
        '✘': '[ ]',  # Heavy X mark
    }
    
    for symbol, replacement in checkbox_replacements.items():
        text = text.replace(symbol, replacement)
    
    # Convert special Unicode symbols to HTML entities
    special_symbols = {
        '©': '&copy;',
        '®': '&reg;',
        '™': '&trade;',
        '°': '&deg;',
        '&': '&amp;',
        '×': '&times;',
        '÷': '&divide;',
        '±': '&plusmn;',
        '≠': '&ne;',
        '≤': '&le;',
        '≥': '&ge;',
        '≈': '&asymp;',
        '≡': '&equiv;',
        'µ': '&micro;',
        '¢': '&cent;',
        '£': '&pound;',
        '¥': '&yen;',
        '€': '&euro;',
        '§': '&sect;',
        '¶': '&para;',
        '•': '&bull;',
        '◊': '&loz;',
        '○': '&cir;',
        '◆': '&diam;',
        '←': '&larr;',
        '→': '&rarr;',
        '↑': '&uarr;',
        '↓': '&darr;',
        '↔': '&harr;',
        '½': '&frac12;',
        '¼': '&frac14;',
        '¾': '&frac34;',
        '¹': '&sup1;',
        '²': '&sup2;',
        '³': '&sup3;',
        '†': '&dagger;',
        '‡': '&Dagger;',
        'α': '&alpha;',
        'β': '&beta;',
        'γ': '&gamma;',
        'δ': '&delta;',
        'π': '&pi;',
        'Σ': '&Sigma;',
        'Ω': '&Omega;',
    }
    
    # Apply replacements (& should be first to avoid double-encoding)
    # But we need to be careful not to replace & in already-encoded entities
    # So we'll do & last and skip already encoded patterns
    amp_handled = False
    for symbol, entity in special_symbols.items():
        if symbol == '&':
            amp_handled = True
            continue
        text = text.replace(symbol, entity)
    
    # Handle & last - only replace standalone & not already part of entities
    if amp_handled:
        # Replace & that's not followed by word chars and semicolon (not already an entity)
        text = re.sub(r'&(?![a-zA-Z0-9#]+;)', '&amp;', text)
    text = text.replace("## ", "").replace("##", "")
    processed_text = text.strip()
    return processed_text

def process_image_with_vllm(image_path: str) -> str:
    """Process a single image with vLLM (using merged model)."""
    global llm
    
    if llm is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")
    
    # Prepare the prompt
    instruction = get_instruction_prompt()
    
    # Create the message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0001,
        top_p=0.8,
        max_tokens=8192,
        min_p=0.05,
        repetition_penalty=1.1
    )
    
    # Generate (no LoRA request needed - weights already merged)
    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
    )
    
    # Extract the generated text
    generated_text = outputs[0].outputs[0].text
    generated_text = postprocess_markdown(generated_text)
    return generated_text


def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path."""
    try:
        # Create a temporary file with the same extension
        suffix = Path(upload_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = upload_file.file.read()
            tmp_file.write(content)
            tmp_file.flush()
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving upload file: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    # Configure via environment variables
    model_path = os.getenv("MODEL_PATH", DEFAULT_MERGED_MODEL)
    tensor_parallel = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    
    success = init_model(model_path, tensor_parallel)
    if not success:
        logger.error("Failed to initialize model on startup!")
        raise RuntimeError("Model initialization failed")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": model_name,
        "model_type": "merged (LoRA weights integrated)",
        "message": "Qwen3-VL Document Parser API is running"
    }


@app.post("/parse", response_model=ParseResponse)
async def parse_document(file: UploadFile = File(...)):
    """
    Parse a PDF or image file and return markdown with tables as HTML.
    
    Args:
        file: Uploaded PDF or image file (png, jpg, jpeg)
    
    Returns:
        JSON response with status and markdown content
    """
    temp_files = []
    
    try:
        # Validate file type
        filename = file.filename.lower()
        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
        
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                }
            )
        
        # Save the uploaded file
        file_path = save_upload_file(file)
        temp_files.append(file_path)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Process based on file type
        if filename.endswith('.pdf'):
            # Convert PDF to images
            images = convert_pdf_to_images(file_path)
            logger.info(f"Converted PDF to {len(images)} images")
            
            # Process each page and combine results
            all_markdowns = []
            for idx, img in enumerate(images):
                # Save image temporarily
                img_temp_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.png'
                ).name
                temp_files.append(img_temp_path)
                img.save(img_temp_path, 'PNG')
                
                logger.info(f"Processing page {idx + 1}/{len(images)}")
                markdown = process_image_with_vllm(img_temp_path)
                
                # Add page separator if multiple pages
                if len(images) > 1:
                    all_markdowns.append(f"{markdown}")
                else:
                    all_markdowns.append(markdown)
            
            # Combine all pages
            final_markdown = "\n\n".join(all_markdowns)
        
        else:
            # Process image directly
            # Verify it's a valid image
            try:
                img = Image.open(file_path)
                img.verify()
                # Reopen after verify
                img = Image.open(file_path)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    rgb_temp_path = tempfile.NamedTemporaryFile(
                        delete=False, suffix='.png'
                    ).name
                    temp_files.append(rgb_temp_path)
                    img.save(rgb_temp_path, 'PNG')
                    file_path = rgb_temp_path
                
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Invalid image file: {str(e)}"
                    }
                )
            
            logger.info(f"Processing image: {file.filename}")
            final_markdown = process_image_with_vllm(file_path)
        
        logger.info("Processing completed successfully")
        
        return ParseResponse(
            status="success",
            markdown=final_markdown
        )
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Processing failed: {str(e)}"
            }
        )
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-VL Document Parser API")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MERGED_MODEL,
        help=f"Path to the merged model (default: {DEFAULT_MERGED_MODEL})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU"
    )
    
    args = parser.parse_args()
    
    # Set environment variables for startup event
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)
    
    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )