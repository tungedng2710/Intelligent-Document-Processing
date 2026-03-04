"""
Batch inference client that calls the existing API endpoint.
Processes images in a folder and saves results as markdown files.
"""
import os
import argparse
import logging
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_image_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """Get all image files from input directory."""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf']
    
    input_path = Path(input_dir)
    image_files = []
    
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def parse_document_via_api(image_path: Path, api_url: str, timeout: int = 300) -> str:
    """Call the API to parse a document."""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            response = requests.post(
                f"{api_url}/parse",
                files=files,
                timeout=timeout
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                return result.get('markdown', '')
            else:
                raise Exception(f"API error: {result.get('message', 'Unknown error')}")
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    except requests.exceptions.Timeout:
        raise Exception(f"Request timeout after {timeout} seconds")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Connection error - is the API running at {api_url}?")
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")


def check_api_health(api_url: str) -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"API Status: {data.get('status')}")
            logger.info(f"Model: {data.get('model')}")
            logger.info(f"LoRA: {data.get('lora_adapter', 'None')}")
            return True
        return False
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        return False


def process_folder(
    input_dir: str,
    output_dir: str,
    api_url: str = "http://localhost:8000",
    timeout: int = 300,
    extensions: List[str] = None
):
    """Process all images in a folder via API and save results as markdown files."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")
    
    # Check API health
    logger.info(f"Checking API at {api_url}...")
    if not check_api_health(api_url):
        logger.error(f"API is not accessible at {api_url}")
        logger.error("Please ensure the API server is running:")
        logger.error("  python services/Qwen3-VL/api_vllm.py")
        return
    
    # Get all image files
    image_files = get_image_files(input_dir, extensions)
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return
    
    # Filter out already processed files
    pending_files = []
    skipped_count = 0
    for image_file in image_files:
        output_filename = f"{image_file.stem}.md"
        output_file = output_path / output_filename
        if output_file.exists():
            skipped_count += 1
        else:
            pending_files.append(image_file)
    
    logger.info(f"Found {len(image_files)} total files")
    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already processed files")
    logger.info(f"Processing {len(pending_files)} remaining files")
    logger.info("")
    
    if not pending_files:
        logger.info("All files already processed. Nothing to do.")
        return
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for image_file in tqdm(pending_files, desc="Processing images"):
        try:
            logger.info(f"Processing: {image_file.name}")
            
            # Call API to parse document
            markdown_content = parse_document_via_api(image_file, api_url, timeout)
            
            # Save markdown file with the same name as input file
            output_filename = f"{image_file.stem}.md"
            output_file = output_path / output_filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"✓ Saved: {output_file.name}\n")
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ Error processing {image_file.name}: {str(e)}\n")
            error_count += 1
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {success_count}/{len(pending_files)}")
    if skipped_count > 0:
        logger.info(f"Skipped (already exists): {skipped_count}")
    if error_count > 0:
        logger.info(f"Errors: {error_count}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference client - processes images via API"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for markdown files"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API endpoint URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf'],
        help="File extensions to process"
    )
    
    args = parser.parse_args()
    
    # Process folder
    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_url=args.api_url,
        timeout=args.timeout,
        extensions=args.extensions
    )


if __name__ == "__main__":
    main()
