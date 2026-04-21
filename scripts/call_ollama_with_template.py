import argparse
import json
import base64
from pathlib import Path
import requests

parser = argparse.ArgumentParser(description="Call Ollama with a prompt template and image.")
parser.add_argument("--prompt-template", required=True, help="Path to the prompt template .txt file")
parser.add_argument("--json-template", required=True, help="Path to the JSON template file")
parser.add_argument("--image", required=True, help="Path to the input image file")
parser.add_argument("--output-dir", required=True, help="Directory to save the output JSON")
parser.add_argument("--model", default="gemma4:e4b-it-bf16", help="Ollama model name")
parser.add_argument("--host", default="http://0.0.0.0:7860", help="Ollama host URL")
args = parser.parse_args()

# --- Load inputs ---
prompt_template = Path(args.prompt_template).read_text()
json_template   = json.load(open(args.json_template))
image_path      = Path(args.image)

# --- Inject template into prompt ---
prompt = prompt_template.replace(
    "{json_template}",
    json.dumps(json_template, ensure_ascii=False, indent=2)
)

# --- Encode image ---
image_b64 = base64.b64encode(image_path.read_bytes()).decode()

# --- Call Ollama ---
payload = {
    "model": args.model,
    "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    "stream": False,
    "think": False,
    "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "think": False,
        },  # low temp for structured extraction
}

resp = requests.post(f"{args.host}/api/chat", json=payload, timeout=500)
resp.raise_for_status()

content = resp.json()["message"]["content"]

# --- Parse JSON output ---
try:
    result = json.loads(content)
except json.JSONDecodeError:
    # Strip markdown code fences if present
    lines = content.strip().split("\n")
    result = json.loads("\n".join(lines[1:-1]))

output_path = Path(args.output_dir) / image_path.with_suffix(".json").name
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f"Saved to {output_path}")
print(json.dumps(result, ensure_ascii=False, indent=2))