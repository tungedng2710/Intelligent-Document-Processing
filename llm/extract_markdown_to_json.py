#!/usr/bin/env python3
"""
Simple script to extract information from markdown files using Ollama LLM.
"""
import requests
import json
import argparse
from pathlib import Path


def read_file(file_path):
    """Read content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def call_ollama(prompt, model="qwen3-vl:8b", api_url="http://0.0.0.0:7860"):
    """Call Ollama API with the given prompt."""
    endpoint = f"{api_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        # Try to get response from 'response' field first, then 'thinking' field
        response_text = result.get('response', '') or result.get('thinking', '')
        
        if not response_text:
            print(f"Warning: Empty response from API")
            print(f"Full API response: {result}")
            raise ValueError("Empty response from Ollama API")
        
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        raise
    except Exception as e:
        print(f"Error processing API response: {e}")
        raise


def extract_markdown_to_json(markdown_path, prompt_path, output_path=None, keys=None):
    """Extract information from markdown file using LLM."""
    # Read the prompt template and markdown content
    prompt_template = read_file(prompt_path)
    markdown_content = read_file(markdown_path)
    
    # If specific keys are provided, modify the prompt to extract only those keys
    if keys:
        keys_list = [k.strip() for k in keys.split(',')]
        schema_fields = ',\n  '.join([f'"{key}": string | null' for key in keys_list])
        custom_schema = '{\n  ' + schema_fields + '\n}'
        
        # Create a custom prompt with specified keys
        prompt_template = f"""You are an information extraction engine.

Task:
Extract structured data from the Markdown text below and return ONLY a valid JSON object.

Rules:
- Output must be strict JSON (no markdown, no code fences, no comments).
- Use null for missing fields.
- Do not invent information.
- Keep numbers as numbers, booleans as booleans, dates as ISO-8601 strings if possible.

Schema to produce:
{custom_schema}

Markdown input:
<<<MARKDOWN
{{INPUT_MARKDOWN}}
MARKDOWN>>>
"""
    
    # Replace placeholder in prompt with actual markdown content
    prompt = prompt_template.replace("{INPUT_MARKDOWN}", markdown_content)
    
    # Call Ollama API
    print(f"Calling Ollama API...")
    response = call_ollama(prompt)
    
    # Parse JSON response
    try:
        result = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON response.")
        print(f"Error: {e}")
        print(f"\nRaw response from API:")
        print(response[:1000])  # Print first 1000 chars
        raise
    
    # Save to output file or print
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved to: {output_path}")
    else:
        print("\nExtracted JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured information from markdown files using Ollama LLM"
    )
    parser.add_argument(
        "markdown_file",
        help="Path to the markdown file to process"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="llm/prompts/extract_json.txt",
        help="Path to prompt template file (default: llm/prompts/extract_json.txt)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output JSON file (optional)"
    )
    parser.add_argument(
        "-m", "--model",
        default="qwen3-vl:8b",
        help="Ollama model to use (default: qwen3-vl:8b)"
    )
    parser.add_argument(
        "-u", "--url",
        default="http://0.0.0.0:7860",
        help="Ollama API URL (default: http://0.0.0.0:7860)"
    )
    parser.add_argument(
        "-k", "--keys",
        help="Comma-separated list of keys to extract (e.g., 'title,author,date'). If not specified, uses default schema from prompt."
    )
    
    args = parser.parse_args()
    
    try:
        extract_markdown_to_json(
            markdown_path=args.markdown_file,
            prompt_path=args.prompt,
            output_path=args.output,
            keys=args.keys
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
