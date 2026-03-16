"""JSON parsing utilities shared across the pipeline."""

from __future__ import annotations

import json
import re


def parse_json_response(text: str) -> dict | list | None:
    """Try to parse JSON from LLM response with multiple fallback strategies."""
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Extract from ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Find first { ... } or [ ... ]
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

    return None
