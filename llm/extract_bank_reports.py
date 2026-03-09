"""
Extract bank report with large vision language model
Prompt: prompts/bank_report_ver_1.0.txt
Engine: Qwen3-VL with Ollama API (0.0.0.0:7860)
API Server: FastAPI with image upload on port 7877
"""
#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from ollama_api import ollama_stream_inference

# Initialize FastAPI app
app = FastAPI(title="Bank Report Extraction API", version="1.0.0")


def read_prompt_template(prompt_path: str) -> str:
    """
    Read the prompt template from a file.
    
    Args:
        prompt_path: Path to the prompt template file
        
    Returns:
        The prompt template as a string
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_bank_report(
    image_path: str,
    prompt_path: str = "llm/prompts/bank_report_ver_1.0.txt",
    model: str = "qwen3-vl",
    api_url: str = "http://116.103.227.252:7860/api/generate",
    output_path: Optional[str] = None
) -> dict:
    """
    Extract structured information from a bank report image using VLM.
    
    Args:
        image_path: Path to the bank report image
        prompt_path: Path to the prompt template file
        model: Name of the Ollama model to use
        api_url: URL of the Ollama API endpoint
        output_path: Optional path to save the JSON output
        
    Returns:
        Extracted information as a dictionary
    """
    # Validate image path
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read prompt template
    print(f"Reading prompt template from: {prompt_path}")
    prompt_template = read_prompt_template(prompt_path)
    
    # Call Ollama API with vision model
    print(f"Processing image: {image_path}")
    print(f"Using model: {model}")
    print(f"Calling API at: {api_url}")
    print("\nExtracting information...\n")
    print("=" * 80)
    
    response = ollama_stream_inference(
        prompt=prompt_template,
        model=model,
        url=api_url,
        image_path=image_path
    )
    
    print("\n" + "=" * 80)
    print("\nExtraction complete.\n")
    
    # Parse JSON response
    try:
        # Try to extract JSON from the response
        # Sometimes the model might wrap JSON in markdown code blocks
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        else:
            json_str = response.strip()
        
        result = json.loads(json_str)
        
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Failed to parse JSON response.")
        print(f"JSON Error: {e}")
        print(f"\n[DEBUG] Raw response (first 2000 chars):")
        print(response[:2000])
        
        # Try to save the raw response for debugging
        if output_path:
            error_path = Path(output_path).with_suffix('.error.txt')
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"\n[INFO] Raw response saved to: {error_path}")
        
        raise
    
    # Save to output file if specified
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Result saved to: {output_path}")
    else:
        print("\n[OUTPUT] Extracted JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


@app.post("/parse")
async def parse_bank_report(
    file: UploadFile = File(...),
    prompt_path: str = "llm/prompts/bank_report_ver_1.0.txt",
    model: str = "qwen3-vl:8b",
    api_url: str = "http://0.0.0.0:7860/api/generate"
):
    """
    API endpoint to parse bank report from uploaded image.
    
    Args:
        file: Uploaded image file
        prompt_path: Path to the prompt template file
        model: Name of the Ollama model to use
        api_url: URL of the Ollama API endpoint
        
    Returns:
        JSON response with 'extract' field containing the extracted data as JSON string
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file to temporary location
    temp_file = None
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix if file.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract information from the image
        result = extract_bank_report(
            image_path=temp_path,
            prompt_path=prompt_path,
            model=model,
            api_url=api_url,
            output_path=None
        )
        
        # Convert result to JSON string for the 'extract' field
        extract_json_str = json.dumps(result, ensure_ascii=False)
        
        # Return response with 'extract' field
        return JSONResponse(content={
            "extract": extract_json_str,
            "status": "success"
        })
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse extraction result: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def start_api_server(host: str = "0.0.0.0", port: int = 7877):
    """
    Start the FastAPI server.
    
    Args:
        host: Host address to bind to
        port: Port number to bind to
    """
    print(f"Starting Bank Report Extraction API server on {host}:{port}")
    print(f"API endpoint: http://{host}:{port}/parse")
    print(f"Documentation: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


def process_batch(
    input_dir: str,
    output_dir: str,
    prompt_path: str,
    model: str,
    api_url: str,
    pattern: str = "*.png"
) -> None:
    """
    Process multiple bank report images in batch.
    
    Args:
        input_dir: Directory containing bank report images
        output_dir: Directory to save extracted JSON files
        prompt_path: Path to the prompt template file
        model: Name of the Ollama model to use
        api_url: URL of the Ollama API endpoint
        pattern: File pattern to match (default: *.png)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching images
    image_files = list(input_path.glob(pattern))
    if not image_files:
        print(f"No images found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_path}")
    print()
    
    # Process each image
    successful = 0
    failed = 0
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing [{idx}/{len(image_files)}]: {image_file.name}")
        print(f"{'=' * 80}\n")
        
        output_file = output_path / f"{image_file.stem}.json"
        
        try:
            extract_bank_report(
                image_path=str(image_file),
                prompt_path=prompt_path,
                model=model,
                api_url=api_url,
                output_path=str(output_file)
            )
            successful += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process {image_file.name}: {e}")
            failed += 1
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Batch processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_files)}")
    print(f"{'=' * 80}\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract structured information from bank report images using Vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start API server
  python extract_bank_reports.py --server
  
  # Start API server on custom host/port
  python extract_bank_reports.py --server --host 0.0.0.0 --port 7877
  
  # Process a single image
  python extract_bank_reports.py input.png -o output.json
  
  # Process a single image with custom prompt
  python extract_bank_reports.py input.png -p custom_prompt.txt -o output.json
  
  # Process multiple images in batch
  python extract_bank_reports.py --batch-input data/bank_reports/ --batch-output results/
  
  # Process with custom model
  python extract_bank_reports.py input.png -m qwen3.5:27b -o output.json
        """
    )
    
    # API server mode
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start API server mode"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7877,
        help="API server port (default: 7877)"
    )
    
    # Single file processing arguments
    parser.add_argument(
        "image",
        nargs='?',
        help="Path to the bank report image file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output JSON file (optional)"
    )
    
    # Batch processing arguments
    parser.add_argument(
        "--batch-input",
        help="Directory containing multiple bank report images for batch processing"
    )
    parser.add_argument(
        "--batch-output",
        help="Directory to save batch processing results"
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="File pattern for batch processing (default: *.png)"
    )
    
    # Common arguments
    parser.add_argument(
        "-p", "--prompt",
        default="llm/prompts/bank_report_ver_1.0.txt",
        help="Path to prompt template file (default: llm/prompts/bank_report_ver_1.0.txt)"
    )
    parser.add_argument(
        "-m", "--model",
        default="qwen3-vl:8b",
        help="Ollama model to use (default: qwen3-vl:8b)"
    )
    parser.add_argument(
        "-u", "--url",
        default="http://0.0.0.0:7860/api/generate",
        help="Ollama API URL (default: http://0.0.0.0:7860/api/generate)"
    )
    
    args = parser.parse_args()
    
    # API server mode
    if args.server:
        start_api_server(host=args.host, port=args.port)
        return 0
    
    # Determine processing mode
    if args.batch_input:
        # Batch processing mode
        if not args.batch_output:
            print("[ERROR] --batch-output is required when using --batch-input")
            return 1
        
        try:
            process_batch(
                input_dir=args.batch_input,
                output_dir=args.batch_output,
                prompt_path=args.prompt,
                model=args.model,
                api_url=args.url,
                pattern=args.pattern
            )
        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")
            return 1
    
    elif args.image:
        # Single file processing mode
        try:
            extract_bank_report(
                image_path=args.image,
                prompt_path=args.prompt,
                model=args.model,
                api_url=args.url,
                output_path=args.output
            )
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return 1
    
    else:
        # No valid input provided
        print("[ERROR] Either provide an image file or use --batch-input for batch processing")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())