import argparse
import base64
import json
from pathlib import Path
import requests

parser = argparse.ArgumentParser(description="Call vLLM with a prompt template and image.")
parser.add_argument("--prompt-template", required=True, help="Path to the prompt template .txt file")
parser.add_argument("--json-template", required=True, help="Path to the JSON template file")
parser.add_argument("--image", required=True, help="Path to the input image file")
parser.add_argument("--output-dir", required=True, help="Directory to save the output JSON")
parser.add_argument("--model", default="Qwen3.5-9B", help="vLLM model name")
parser.add_argument("--no-think", action="store_true", help="Disable vLLM thinking mode if supported")
parser.add_argument("--host", default="http://localhost:9888", help="vLLM host URL")
args = parser.parse_args()

# --- Load inputs ---
prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")
json_template = json.loads(Path(args.json_template).read_text(encoding="utf-8"))
image_path = Path(args.image)

# --- Inject template into prompt ---
prompt = prompt_template.replace(
    "{json_template}",
    json.dumps(json_template, ensure_ascii=False, indent=2)
)

# --- Encode image ---
image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
mime_type = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}.get(image_path.suffix.lower(), "image/png")

payload = {
    "model": args.model,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ],
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 12000,
}
if args.no_think:
    payload["think"] = False

base_url = args.host.rstrip("/")
resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=500)
resp.raise_for_status()
result = resp.json()
content = ""
choices = result.get("choices", [])
if choices:
    content = choices[0].get("message", {}).get("content", "")

# --- Parse JSON output ---
parsed = None
try:
    parsed = json.loads(content)
except (json.JSONDecodeError, TypeError):
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            parsed = None

output = {
    "image": str(image_path.name),
    "model": args.model,
    "content": content,
}
if parsed is not None:
    output["parsed_json"] = parsed

output_path = Path(args.output_dir) / image_path.with_suffix(".json").name
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved to {output_path}")
print(json.dumps(output, ensure_ascii=False, indent=2))
