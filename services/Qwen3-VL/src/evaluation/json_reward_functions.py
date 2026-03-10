"""
JSON-specific Reward Functions for Bank Report Document Extraction (GRPO)
=========================================================================

Reward functions designed for evaluating JSON output quality against
ground truth JSON annotations. Used with GRPO training where the model
outputs structured JSON (not tagged blocks).

Reward functions:
1. json_validity_reward       - Valid JSON parsing check
2. json_key_matching_reward   - Key structure matching (recursive)
3. json_value_similarity_reward - Value content similarity (NED + Jaccard)
4. json_structure_reward      - Combined structural + content score
"""

import re
import json
import math
from typing import List, Any, Dict, Tuple, Optional, Union
from difflib import SequenceMatcher

# =============================================================================
# HELPER: Extract text from GRPO completion format
# =============================================================================

def extract_text_from_completion(completion: Any) -> str:
    """Extract text content from a completion (string, list, or dict)."""
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list):
        text_parts = []
        for part in completion:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if 'text' in part:
                    text_parts.append(part['text'])
                elif 'content' in part:
                    content = part['content']
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, str):
                                text_parts.append(c)
                            elif isinstance(c, dict) and 'text' in c:
                                text_parts.append(c['text'])
        return '\n'.join(text_parts)
    elif isinstance(completion, dict):
        if 'text' in completion:
            return completion['text']
        elif 'content' in completion:
            return extract_text_from_completion(completion['content'])
    return str(completion)


# =============================================================================
# JSON PARSING UTILITIES
# =============================================================================

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON object from text that may contain markdown fences or extra text.

    Tries:
    1. Direct json.loads
    2. Extract from ```json ... ``` fences
    3. Find first { ... } balanced block
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Extract from markdown code fence
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Find first balanced { ... }
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, TypeError):
                        break

    return None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalized_edit_distance(text1: str, text2: str) -> float:
    """Compute normalized edit distance similarity (0 to 1)."""
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return SequenceMatcher(None, t1, t2).ratio()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity on word tokens."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def combined_text_similarity(text1: str, text2: str) -> float:
    """Combined NED + Jaccard similarity."""
    ned = normalized_edit_distance(text1, text2)
    jac = jaccard_similarity(text1, text2)
    return 0.6 * ned + 0.4 * jac


# =============================================================================
# KEY EXTRACTION (RECURSIVE)
# =============================================================================

def flatten_keys(obj: Any, prefix: str = "") -> List[str]:
    """Recursively extract all key paths from a JSON object."""
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.append(full_key)
            keys.extend(flatten_keys(v, full_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            keys.extend(flatten_keys(item, f"{prefix}[{i}]"))
    return keys


def flatten_key_set(obj: Any) -> set:
    """Get set of normalized top-level and nested key names (without indices)."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(normalize_text(k))
            if isinstance(v, dict):
                for sk in flatten_key_set(v):
                    keys.add(sk)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for sk in flatten_key_set(item):
                            keys.add(sk)
    return keys


# =============================================================================
# VALUE EXTRACTION (RECURSIVE)
# =============================================================================

def flatten_values(obj: Any) -> List[str]:
    """Recursively extract all leaf values from a JSON object as strings."""
    values = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(flatten_values(v))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(flatten_values(item))
    elif obj is not None:
        values.append(str(obj))
    return values


def flatten_kv_pairs(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """Recursively extract all (key_path, value) pairs from a JSON object."""
    pairs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                pairs.extend(flatten_kv_pairs(v, full_key))
            else:
                pairs.append((full_key, str(v) if v is not None else ""))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            pairs.extend(flatten_kv_pairs(item, f"{prefix}[{i}]"))
    return pairs


# =============================================================================
# REWARD FUNCTION 1: JSON Validity
# =============================================================================

def json_validity_reward(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward for generating valid JSON.
    Returns:
        +2.0 if valid JSON object
        +0.5 if valid JSON but not a dict (e.g., list)
        -2.0 if invalid JSON
    """
    rewards = []
    for completion in completions:
        text = extract_text_from_completion(completion)
        parsed = extract_json_from_text(text)
        if parsed is not None:
            if isinstance(parsed, dict):
                rewards.append(2.0)
            else:
                rewards.append(0.5)
        else:
            rewards.append(-2.0)
    return rewards


# =============================================================================
# REWARD FUNCTION 2: Key Matching
# =============================================================================

def json_key_matching_reward(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward for matching JSON key structure against reference.
    Uses F1 score on the set of all key names (top-level + nested).

    Returns reward in [-2.0, 3.0] range:
        - High F1 on key names → positive reward
        - Missing/extra keys → reduced reward
        - Invalid JSON → -2.0
    """
    rewards = []
    for completion, ref in zip(completions, answer):
        text = extract_text_from_completion(completion)
        pred_json = extract_json_from_text(text)

        ref_text = extract_text_from_completion(ref)
        ref_json = extract_json_from_text(ref_text)
        if ref_json is None:
            # If reference can't be parsed (shouldn't happen), try loading directly
            try:
                ref_json = json.loads(ref_text)
            except Exception:
                rewards.append(0.0)
                continue

        if pred_json is None:
            rewards.append(-2.0)
            continue

        pred_keys = flatten_key_set(pred_json)
        ref_keys = flatten_key_set(ref_json)

        if not ref_keys:
            rewards.append(0.0)
            continue

        tp = len(pred_keys & ref_keys)
        fp = len(pred_keys - ref_keys)
        fn = len(ref_keys - pred_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Scale F1 [0,1] → [-1.0, 3.0]
        reward = f1 * 4.0 - 1.0
        rewards.append(reward)

    return rewards


# =============================================================================
# REWARD FUNCTION 3: Value Similarity
# =============================================================================

def json_value_similarity_reward(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward for value content accuracy.
    Compares all leaf values using combined text similarity.

    Returns reward in [-2.0, 3.0] range.
    """
    rewards = []
    for completion, ref in zip(completions, answer):
        text = extract_text_from_completion(completion)
        pred_json = extract_json_from_text(text)

        ref_text = extract_text_from_completion(ref)
        ref_json = extract_json_from_text(ref_text)
        if ref_json is None:
            try:
                ref_json = json.loads(ref_text)
            except Exception:
                rewards.append(0.0)
                continue

        if pred_json is None:
            rewards.append(-2.0)
            continue

        # Get all key-value pairs
        pred_kv = flatten_kv_pairs(pred_json)
        ref_kv = flatten_kv_pairs(ref_json)

        if not ref_kv:
            rewards.append(0.0)
            continue

        # Match values by normalized key names using greedy matching
        pred_by_key = {}
        for k, v in pred_kv:
            nk = normalize_text(k.split('.')[-1].rstrip(']').split('[')[0])
            pred_by_key.setdefault(nk, []).append(v)

        ref_by_key = {}
        for k, v in ref_kv:
            nk = normalize_text(k.split('.')[-1].rstrip(']').split('[')[0])
            ref_by_key.setdefault(nk, []).append(v)

        total_sim = 0.0
        count = 0

        for key, ref_vals in ref_by_key.items():
            pred_vals = pred_by_key.get(key, [])
            for i, rv in enumerate(ref_vals):
                if i < len(pred_vals):
                    sim = combined_text_similarity(pred_vals[i], rv)
                else:
                    sim = 0.0  # Missing value
                total_sim += sim
                count += 1

        avg_sim = total_sim / count if count > 0 else 0.0

        # Scale [0,1] → [-1.0, 3.0]
        reward = avg_sim * 4.0 - 1.0
        rewards.append(reward)

    return rewards


# =============================================================================
# REWARD FUNCTION 4: Combined JSON Structure Reward
# =============================================================================

def json_structure_reward(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    Combined structural + content score for JSON outputs.
    Evaluates:
    - Key hierarchy correctness (nesting depth matches)
    - Array structure preservation (tables as arrays)
    - Overall content fidelity

    Returns reward in [-3.0, 4.0] range.
    """
    rewards = []
    for completion, ref in zip(completions, answer):
        text = extract_text_from_completion(completion)
        pred_json = extract_json_from_text(text)

        ref_text = extract_text_from_completion(ref)
        ref_json = extract_json_from_text(ref_text)
        if ref_json is None:
            try:
                ref_json = json.loads(ref_text)
            except Exception:
                rewards.append(0.0)
                continue

        if pred_json is None:
            rewards.append(-3.0)
            continue

        score = 0.0

        # --- Component 1: Top-level key matching (max 1.5) ---
        ref_top_keys = set(normalize_text(k) for k in ref_json.keys()) if isinstance(ref_json, dict) else set()
        pred_top_keys = set(normalize_text(k) for k in pred_json.keys()) if isinstance(pred_json, dict) else set()

        if ref_top_keys:
            tp = len(ref_top_keys & pred_top_keys)
            recall = tp / len(ref_top_keys)
            precision = tp / len(pred_top_keys) if pred_top_keys else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            score += f1 * 1.5

        # --- Component 2: Nested structure match (max 1.0) ---
        def count_nested_types(obj):
            dicts, lists, leaves = 0, 0, 0
            if isinstance(obj, dict):
                dicts += 1
                for v in obj.values():
                    d, l, le = count_nested_types(v)
                    dicts += d
                    lists += l
                    leaves += le
            elif isinstance(obj, list):
                lists += 1
                for item in obj:
                    d, l, le = count_nested_types(item)
                    dicts += d
                    lists += l
                    leaves += le
            else:
                leaves += 1
            return dicts, lists, leaves

        ref_d, ref_l, ref_le = count_nested_types(ref_json)
        pred_d, pred_l, pred_le = count_nested_types(pred_json)

        # Compare structure proportions
        struct_sim = 0.0
        total_ref = ref_d + ref_l + ref_le
        total_pred = pred_d + pred_l + pred_le

        if total_ref > 0 and total_pred > 0:
            ref_ratios = [ref_d / total_ref, ref_l / total_ref, ref_le / total_ref]
            pred_ratios = [pred_d / total_pred, pred_l / total_pred, pred_le / total_pred]
            # Cosine-like similarity between structure ratios
            dot = sum(a * b for a, b in zip(ref_ratios, pred_ratios))
            mag_ref = math.sqrt(sum(a * a for a in ref_ratios))
            mag_pred = math.sqrt(sum(a * a for a in pred_ratios))
            if mag_ref > 0 and mag_pred > 0:
                struct_sim = dot / (mag_ref * mag_pred)

        score += struct_sim * 1.0

        # --- Component 3: Value content similarity (max 1.5) ---
        ref_values = flatten_values(ref_json)
        pred_values = flatten_values(pred_json)

        ref_text_concat = " ".join(str(v) for v in ref_values)
        pred_text_concat = " ".join(str(v) for v in pred_values)

        content_sim = combined_text_similarity(ref_text_concat, pred_text_concat)
        score += content_sim * 1.5

        # Scale to [-3.0, 4.0]
        reward = score - 0.5  # Shift so mediocre outputs get ~0
        reward = max(-3.0, min(4.0, reward))
        rewards.append(reward)

    return rewards
