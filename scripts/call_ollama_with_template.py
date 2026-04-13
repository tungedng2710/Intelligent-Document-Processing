import json
import base64
from pathlib import Path
import requests

# --- Load inputs ---
prompt_template = Path("/root/tungn197/idp/data/prompts/healthcare_types/phieu_kham_benh_vao_vien.txt").read_text()
json_template   = json.load(open("/root/tungn197/idp/data/prompts/healthcare_types/phieu_kham_benh_vao_vien_template.json"))
image_path      = Path("/root/tungn197/idp/data/healthcare/hoso1_pages/page-03.png")

# --- Inject template into prompt ---
prompt = prompt_template.replace(
    "{json_template}",
    json.dumps(json_template, ensure_ascii=False, indent=2)
)

# --- Encode image ---
image_b64 = base64.b64encode(image_path.read_bytes()).decode()

# --- Call Ollama ---
payload = {
    "model": "qwen3.5:cloud",
    "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    "stream": False,
    "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
        },   # low temp for structured extraction
}

resp = requests.post("http://0.0.0.0:7860/api/chat", json=payload, timeout=300)
resp.raise_for_status()

content = resp.json()["message"]["content"]

# --- Parse JSON output ---
try:
    result = json.loads(content)
except json.JSONDecodeError:
    # Strip markdown code fences if present
    lines = content.strip().split("\n")
    result = json.loads("\n".join(lines[1:-1]))

output_path = image_path.with_suffix(".json")
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f"Saved to {output_path}")
print(json.dumps(result, ensure_ascii=False, indent=2))