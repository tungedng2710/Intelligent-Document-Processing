"""
Sophisticated Reward Functions for Block Classification
========================================================

This module implements robust reward functions with:
1. Table-First Matching: Focus on matching table blocks first (minority class)
2. Hungarian Algorithm: Optimal bipartite matching for block assignment
3. Multiple Similarity Metrics: NED, Jaccard, token overlap
4. Handling for merged/split blocks
5. Asymmetric penalties for different error types

Author: Expert VLM Fine-tuning
Date: 2026-01-02
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
import html


# =============================================================================
# HELPER FUNCTION FOR GRPO COMPATIBILITY
# =============================================================================

def extract_text_from_completion(completion: Any) -> str:
    """
    Extract text content from a completion which can be:
    - A string (direct text)
    - A list of message parts (chat format from GRPO)
    - A dict with 'content' key
    
    Args:
        completion: The completion in various possible formats
        
    Returns:
        The extracted text as a string
    """
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list):
        # It's a list of message parts - extract text from each
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
# DATA STRUCTURES
# =============================================================================

@dataclass
class Block:
    """Represents a parsed block."""
    block_type: str  # 'text' or 'table'
    content: str     # Raw content including tags
    clean_content: str  # Content with HTML tags stripped
    position: int    # Position in original text


@dataclass 
class MatchResult:
    """Result of matching predicted blocks to reference blocks."""
    matched_pairs: List[Tuple[int, int, float]]  # (pred_idx, ref_idx, similarity)
    unmatched_pred: List[int]  # Predicted blocks not matched
    unmatched_ref: List[int]   # Reference blocks not matched
    total_similarity: float


# =============================================================================
# TEXT CLEANING UTILITIES
# =============================================================================

def strip_html_tags(text: str) -> str:
    """
    Remove all HTML tags from text, keeping only the text content.
    Also normalizes whitespace.
    """
    # Decode HTML entities first
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_table_text(table_html: str) -> str:
    """
    Extract text content from HTML table, preserving cell structure.
    Returns space-separated cell contents.
    """
    # Extract content from td and th tags
    cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', table_html, re.DOTALL | re.IGNORECASE)
    
    # Clean each cell
    cleaned_cells = [strip_html_tags(cell) for cell in cells]
    
    # Remove empty cells
    cleaned_cells = [c for c in cleaned_cells if c.strip()]
    
    return ' '.join(cleaned_cells)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
    - Lowercase
    - Remove punctuation (keep alphanumeric and spaces)
    - Normalize whitespace
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =============================================================================
# SIMILARITY METRICS
# =============================================================================

def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity between two texts based on word sets.
    
    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def normalized_edit_distance(text1: str, text2: str) -> float:
    """
    Compute Normalized Edit Distance (NED) similarity.
    NED = 1 - (levenshtein_distance / max_length)
    
    Uses SequenceMatcher for efficiency on longer texts.
    
    Returns:
        Similarity score between 0 and 1
    """
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    # Use SequenceMatcher ratio (similar to NED but more efficient)
    return SequenceMatcher(None, text1, text2).ratio()


def token_overlap_similarity(text1: str, text2: str, use_idf: bool = False) -> float:
    """
    Compute token overlap with optional IDF weighting.
    
    Args:
        text1: First text
        text2: Second text
        use_idf: Whether to use IDF weighting (requires corpus, simplified here)
    
    Returns:
        Similarity score between 0 and 1
    """
    tokens1 = normalize_text(text1).split()
    tokens2 = normalize_text(text2).split()
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    # Count token occurrences
    from collections import Counter
    count1 = Counter(tokens1)
    count2 = Counter(tokens2)
    
    # Compute overlap (minimum count for each token)
    overlap = sum((count1 & count2).values())
    total = sum(count1.values()) + sum(count2.values())
    
    # Dice coefficient: 2 * overlap / total
    return 2 * overlap / total if total > 0 else 0.0


def combined_similarity(
    text1: str, 
    text2: str,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute combined similarity using multiple metrics.
    
    Args:
        text1: First text
        text2: Second text
        weights: Weights for each metric (default: equal weights)
    
    Returns:
        Weighted average similarity score
    """
    if weights is None:
        weights = {
            'jaccard': 0.3,
            'ned': 0.4,
            'token_overlap': 0.3
        }
    
    scores = {
        'jaccard': jaccard_similarity(text1, text2),
        'ned': normalized_edit_distance(text1, text2),
        'token_overlap': token_overlap_similarity(text1, text2),
    }
    
    total_weight = sum(weights.values())
    weighted_sum = sum(scores[k] * weights.get(k, 0) for k in scores)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


# =============================================================================
# BLOCK PARSING
# =============================================================================

def parse_blocks(text: str) -> List[Block]:
    """
    Parse text into list of Block objects.
    Handles both tagged format and raw markdown with --- separators.
    """
    blocks = []
    
    # Try tagged format first
    text_pattern = r'<text_block>(.*?)</text_block>'
    table_pattern = r'<table_block>(.*?)</table_block>'
    
    for match in re.finditer(text_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        blocks.append(Block(
            block_type='text',
            content=content,
            clean_content=strip_html_tags(content),
            position=match.start()
        ))
    
    for match in re.finditer(table_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        # For tables, extract text from table HTML
        clean = extract_table_text(content) if '<table>' in content.lower() else strip_html_tags(content)
        blocks.append(Block(
            block_type='table',
            content=content,
            clean_content=clean,
            position=match.start()
        ))
    
    # If no tagged blocks found, try --- separator format
    if not blocks:
        parts = text.split('---')
        position = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Determine type based on content
            if '<table>' in part.lower():
                block_type = 'table'
                clean = extract_table_text(part)
            else:
                block_type = 'text'
                clean = strip_html_tags(part)
            
            blocks.append(Block(
                block_type=block_type,
                content=part,
                clean_content=clean,
                position=position
            ))
            position += len(part) + 3  # +3 for ---
    
    # Sort by position
    blocks.sort(key=lambda x: x.position)
    
    return blocks


def get_blocks_by_type(blocks: List[Block]) -> Tuple[List[Block], List[Block]]:
    """Separate blocks into text and table lists."""
    text_blocks = [b for b in blocks if b.block_type == 'text']
    table_blocks = [b for b in blocks if b.block_type == 'table']
    return text_blocks, table_blocks


# =============================================================================
# HUNGARIAN MATCHING
# =============================================================================

def compute_similarity_matrix(
    pred_blocks: List[Block],
    ref_blocks: List[Block],
    similarity_fn: Callable[[str, str], float] = combined_similarity
) -> np.ndarray:
    """
    Compute similarity matrix between predicted and reference blocks.
    
    Args:
        pred_blocks: List of predicted blocks
        ref_blocks: List of reference blocks
        similarity_fn: Function to compute similarity between two texts
    
    Returns:
        Similarity matrix of shape (len(pred_blocks), len(ref_blocks))
    """
    n_pred = len(pred_blocks)
    n_ref = len(ref_blocks)
    
    if n_pred == 0 or n_ref == 0:
        return np.zeros((max(n_pred, 1), max(n_ref, 1)))
    
    sim_matrix = np.zeros((n_pred, n_ref))
    
    for i, pred in enumerate(pred_blocks):
        for j, ref in enumerate(ref_blocks):
            sim_matrix[i, j] = similarity_fn(pred.clean_content, ref.clean_content)
    
    return sim_matrix


def hungarian_match(
    pred_blocks: List[Block],
    ref_blocks: List[Block],
    similarity_fn: Callable[[str, str], float] = combined_similarity,
    threshold: float = 0.3
) -> MatchResult:
    """
    Find optimal matching between predicted and reference blocks using Hungarian algorithm.
    
    Args:
        pred_blocks: List of predicted blocks
        ref_blocks: List of reference blocks
        similarity_fn: Similarity function
        threshold: Minimum similarity for a valid match
    
    Returns:
        MatchResult with matched pairs and unmatched blocks
    """
    n_pred = len(pred_blocks)
    n_ref = len(ref_blocks)
    
    if n_pred == 0 and n_ref == 0:
        return MatchResult([], [], [], 0.0)
    
    if n_pred == 0:
        return MatchResult([], [], list(range(n_ref)), 0.0)
    
    if n_ref == 0:
        return MatchResult([], list(range(n_pred)), [], 0.0)
    
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(pred_blocks, ref_blocks, similarity_fn)
    
    # Convert to cost matrix (Hungarian finds minimum, we want maximum)
    cost_matrix = 1.0 - sim_matrix
    
    # Handle rectangular matrices by padding
    max_size = max(n_pred, n_ref)
    padded_cost = np.ones((max_size, max_size))
    padded_cost[:n_pred, :n_ref] = cost_matrix
    
    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    # Extract valid matches (above threshold)
    matched_pairs = []
    matched_pred = set()
    matched_ref = set()
    total_similarity = 0.0
    
    for pred_idx, ref_idx in zip(row_ind, col_ind):
        if pred_idx < n_pred and ref_idx < n_ref:
            similarity = sim_matrix[pred_idx, ref_idx]
            if similarity >= threshold:
                matched_pairs.append((pred_idx, ref_idx, similarity))
                matched_pred.add(pred_idx)
                matched_ref.add(ref_idx)
                total_similarity += similarity
    
    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_ref = [i for i in range(n_ref) if i not in matched_ref]
    
    return MatchResult(matched_pairs, unmatched_pred, unmatched_ref, total_similarity)


# =============================================================================
# TABLE-FIRST MATCHING REWARD
# =============================================================================

def table_first_matching_reward(
    prediction: str,
    reference: str,
    config: Dict = None
) -> Dict[str, float]:
    """
    Compute reward using Table-First Matching strategy.
    
    Strategy:
    1. Parse blocks from both prediction and reference
    2. Match table blocks first using Hungarian algorithm
    3. Analyze table matching results for classification accuracy
    4. Match remaining text blocks
    5. Compute comprehensive reward
    
    Args:
        prediction: Model output
        reference: Ground truth
        config: Reward configuration
    
    Returns:
        Dictionary with detailed reward breakdown
    """
    if config is None:
        config = {
            # Matching thresholds
            'table_match_threshold': 0.25,  # Lower threshold for tables (may have different structures)
            'text_match_threshold': 0.3,
            
            # Rewards
            'correct_table_match': 3.0,      # Correctly classified table
            'correct_text_match': 1.0,       # Correctly classified text
            'partial_table_match': 1.5,      # Table matched but low similarity
            
            # Penalties
            'table_as_text_penalty': -4.0,   # Table in ref not matched (missed as text)
            'text_as_table_penalty': -2.0,   # Extra table in pred (text classified as table)
            'missing_block_penalty': -0.5,   # Block in ref not found at all
            'extra_block_penalty': -0.3,     # Extra block in pred
            
            # Bonuses
            'table_count_match_bonus': 1.0,  # Same number of tables
            'high_similarity_bonus': 0.5,    # Per match with similarity > 0.7
        }
    
    # Parse blocks
    pred_blocks = parse_blocks(prediction)
    ref_blocks = parse_blocks(reference)
    
    pred_text, pred_tables = get_blocks_by_type(pred_blocks)
    ref_text, ref_tables = get_blocks_by_type(ref_blocks)
    
    # Initialize reward components
    rewards = {
        'table_classification': 0.0,
        'text_classification': 0.0,
        'content_quality': 0.0,
        'structure_quality': 0.0,
        'penalties': 0.0,
        'bonuses': 0.0,
    }
    
    # =========================================================================
    # Step 1: Match Table Blocks (Priority)
    # =========================================================================
    
    table_match = hungarian_match(
        pred_tables, 
        ref_tables,
        similarity_fn=combined_similarity,
        threshold=config['table_match_threshold']
    )
    
    # Reward for correctly matched tables
    for pred_idx, ref_idx, similarity in table_match.matched_pairs:
        if similarity >= 0.5:
            rewards['table_classification'] += config['correct_table_match']
        else:
            rewards['table_classification'] += config['partial_table_match']
        
        rewards['content_quality'] += similarity * config['correct_table_match'] * 0.5
        
        if similarity >= 0.7:
            rewards['bonuses'] += config['high_similarity_bonus']
    
    # Penalty for unmatched reference tables (Table→Text errors)
    # These are tables in the reference that weren't matched - model probably output them as text
    for ref_idx in table_match.unmatched_ref:
        ref_table = ref_tables[ref_idx]
        
        # Check if this table content appears in predicted text blocks
        found_in_text = False
        for pred_text_block in pred_text:
            sim = combined_similarity(ref_table.clean_content, pred_text_block.clean_content)
            if sim >= 0.3:
                # Table content found in text block - clear misclassification
                rewards['penalties'] += config['table_as_text_penalty']
                found_in_text = True
                break
        
        if not found_in_text:
            # Table content missing entirely
            rewards['penalties'] += config['missing_block_penalty'] * 2
    
    # Penalty for extra predicted tables (Text→Table errors)
    # These are tables in prediction not matched to any reference table
    for pred_idx in table_match.unmatched_pred:
        pred_table = pred_tables[pred_idx]
        
        # Check if this table content matches any reference text block
        matched_to_text = False
        for ref_text_block in ref_text:
            sim = combined_similarity(pred_table.clean_content, ref_text_block.clean_content)
            if sim >= 0.3:
                # Table content matches text block - misclassification
                rewards['penalties'] += config['text_as_table_penalty']
                matched_to_text = True
                break
        
        if not matched_to_text:
            # Hallucinated table?
            rewards['penalties'] += config['extra_block_penalty'] * 2
    
    # Bonus for matching table count
    if len(pred_tables) == len(ref_tables):
        rewards['bonuses'] += config['table_count_match_bonus']
    
    # =========================================================================
    # Step 2: Match Text Blocks
    # =========================================================================
    
    text_match = hungarian_match(
        pred_text,
        ref_text,
        similarity_fn=combined_similarity,
        threshold=config['text_match_threshold']
    )
    
    # Reward for matched text blocks
    for pred_idx, ref_idx, similarity in text_match.matched_pairs:
        rewards['text_classification'] += config['correct_text_match']
        rewards['content_quality'] += similarity * config['correct_text_match'] * 0.3
        
        if similarity >= 0.7:
            rewards['bonuses'] += config['high_similarity_bonus'] * 0.5
    
    # Minor penalty for unmatched text blocks (less severe than table errors)
    rewards['penalties'] += len(text_match.unmatched_ref) * config['missing_block_penalty']
    rewards['penalties'] += len(text_match.unmatched_pred) * config['extra_block_penalty']
    
    # =========================================================================
    # Step 3: Structure Quality
    # =========================================================================
    
    # Check block count similarity
    total_pred = len(pred_blocks)
    total_ref = len(ref_blocks)
    
    if total_ref > 0:
        count_ratio = min(total_pred, total_ref) / max(total_pred, total_ref)
        rewards['structure_quality'] = count_ratio * 1.0
    
    # =========================================================================
    # Compute Total Reward
    # =========================================================================
    
    total_reward = sum(rewards.values())
    
    # Normalize by reference block count for comparability
    if len(ref_blocks) > 0:
        normalized_reward = total_reward / len(ref_blocks)
    else:
        normalized_reward = total_reward
    
    rewards['total'] = total_reward
    rewards['normalized'] = normalized_reward
    rewards['num_pred_blocks'] = len(pred_blocks)
    rewards['num_ref_blocks'] = len(ref_blocks)
    rewards['num_pred_tables'] = len(pred_tables)
    rewards['num_ref_tables'] = len(ref_tables)
    rewards['table_matches'] = len(table_match.matched_pairs)
    rewards['text_matches'] = len(text_match.matched_pairs)
    
    return rewards


# =============================================================================
# GRPO REWARD FUNCTIONS (Compatible with TRL)
# =============================================================================

def block_classification_reward_v2(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    GRPO-compatible reward function using Table-First Matching.
    
    Args:
        prompts: List of prompts (unused)
        completions: List of model completions (can be strings or chat format)
        answer: List of reference answers
    
    Returns:
        List of reward scores
    """
    rewards = []
    
    for completion, reference in zip(completions, answer):
        # Extract text from completion (handles both string and list formats)
        completion_text = extract_text_from_completion(completion)
        reference_text = extract_text_from_completion(reference)
        
        result = table_first_matching_reward(completion_text, reference_text)
        
        # Scale normalized reward to reasonable range [-5, 5]
        scaled_reward = max(-5.0, min(5.0, result['normalized'] * 2))
        rewards.append(scaled_reward)
    
    return rewards


def table_accuracy_reward(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    Focused reward on table classification accuracy only.
    This is the core metric we want to optimize.
    """
    rewards = []
    
    for completion, reference in zip(completions, answer):
        # Extract text from completion (handles both string and list formats)
        completion_text = extract_text_from_completion(completion)
        reference_text = extract_text_from_completion(reference)
        
        pred_blocks = parse_blocks(completion_text)
        ref_blocks = parse_blocks(reference_text)
        
        _, pred_tables = get_blocks_by_type(pred_blocks)
        _, ref_tables = get_blocks_by_type(ref_blocks)
        
        # Match tables
        table_match = hungarian_match(
            pred_tables,
            ref_tables,
            threshold=0.25
        )
        
        n_ref = len(ref_tables)
        n_pred = len(pred_tables)
        n_matched = len(table_match.matched_pairs)
        
        if n_ref == 0 and n_pred == 0:
            # No tables in either - perfect
            rewards.append(1.0)
        elif n_ref == 0:
            # No tables in ref but predicted some - penalty
            rewards.append(-1.0 * n_pred)
        elif n_pred == 0:
            # Tables in ref but none predicted - penalty
            rewards.append(-2.0 * n_ref)
        else:
            # Compute precision and recall
            precision = n_matched / n_pred if n_pred > 0 else 0
            recall = n_matched / n_ref if n_ref > 0 else 0
            
            # F1 score weighted toward recall (catching all tables is more important)
            if precision + recall > 0:
                f_beta = (1 + 1.5**2) * (precision * recall) / (1.5**2 * precision + recall)
            else:
                f_beta = 0
            
            # Scale to [-2, 2]
            reward = f_beta * 4 - 2
            rewards.append(reward)
    
    return rewards


def content_similarity_reward(
    completions: List[Any],
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for overall content similarity (regardless of block classification).
    """
    rewards = []
    
    for completion, reference in zip(completions, answer):
        # Extract text from completion (handles both string and list formats)
        completion_text = extract_text_from_completion(completion)
        reference_text = extract_text_from_completion(reference)
        
        # Strip all tags and compare
        clean_pred = strip_html_tags(re.sub(r'</?(?:text_block|table_block)>', '', completion_text))
        clean_ref = strip_html_tags(re.sub(r'</?(?:text_block|table_block)>', '', reference_text))
        
        similarity = combined_similarity(clean_pred, clean_ref)
        
        # Scale to [-1, 2]
        reward = similarity * 3 - 1
        rewards.append(reward)
    
    return rewards


def format_compliance_reward_v2(
    completions: List[Any],
    **kwargs
) -> List[float]:
    """
    Reward for correct format usage.
    """
    rewards = []
    
    for completion in completions:
        # Extract text from completion (handles both string and list formats)
        completion_text = extract_text_from_completion(completion)
        
        score = 0.0
        
        # Check for block tags
        has_blocks = '<text_block>' in completion_text or '<table_block>' in completion_text
        if has_blocks:
            score += 0.5
        
        # Check balanced tags
        text_opens = completion_text.count('<text_block>')
        text_closes = completion_text.count('</text_block>')
        table_opens = completion_text.count('<table_block>')
        table_closes = completion_text.count('</table_block>')
        
        if text_opens == text_closes and table_opens == table_closes:
            score += 0.5
        else:
            score -= 0.5
        
        # Check table blocks contain <table>
        table_blocks = re.findall(r'<table_block>(.*?)</table_block>', completion_text, re.DOTALL)
        if table_blocks:
            valid_tables = sum(1 for b in table_blocks if '<table>' in b.lower())
            score += 0.5 * (valid_tables / len(table_blocks))
        
        # Check no nested blocks
        if re.search(r'<text_block>.*<text_block>', completion_text, re.DOTALL):
            score -= 0.5
        if re.search(r'<table_block>.*<table_block>', completion_text, re.DOTALL):
            score -= 0.5
        
        rewards.append(max(-1.0, min(1.0, score)))
    
    return rewards


# =============================================================================
# TESTING AND DEBUGGING
# =============================================================================

def test_matching():
    """Test the matching logic with sample data."""
    
    # Sample reference (from sample.md)
    reference = """
<text_block>
## Shipper
BOTOU PANGLONG FRUIT PRODUCTS RESPONSIBILITY CO.,LTD.
WANGWU TOWN BOTOU CITY HEBEI PROVINCE,CHINA
</text_block>
<text_block>
## Booking No.
CN05434968
</text_block>
<table_block>
<table>
<thead><tr><th>Container No.</th><th>No. of P'kgs</th><th>Description</th></tr></thead>
<tbody><tr><td>BMOU9237865</td><td>4868</td><td>FRESH PEAR</td></tr></tbody>
</table>
</table_block>
<text_block>
## Total Number of Containers
SAY : FOUR (4) CONTAINERS ONLY.
</text_block>
"""

    # Sample prediction (with errors)
    prediction_good = """
<text_block>
## Shipper
BOTOU PANGLONG FRUIT PRODUCTS RESPONSIBILITY CO.,LTD.
WANGWU TOWN BOTOU CITY HEBEI PROVINCE,CHINA
</text_block>
<text_block>
## Booking No.
CN05434968
</text_block>
<table_block>
<table>
<tr><th>Container No.</th><th>Packages</th><th>Description</th></tr>
<tr><td>BMOU9237865</td><td>4868 CARTONS</td><td>FRESH PEAR</td></tr>
</table>
</table_block>
<text_block>
## Total Containers
SAY : FOUR (4) CONTAINERS ONLY.
</text_block>
"""

    prediction_bad = """
<text_block>
## Shipper
BOTOU PANGLONG FRUIT PRODUCTS RESPONSIBILITY CO.,LTD.
</text_block>
<table_block>
<table>
<tr><td>Booking No.</td><td>CN05434968</td></tr>
</table>
</table_block>
<text_block>
Container No. BMOU9237865
Packages: 4868
Description: FRESH PEAR
</text_block>
<text_block>
## Total Containers
FOUR CONTAINERS
</text_block>
"""

    print("="*60)
    print("Testing Table-First Matching Reward")
    print("="*60)
    
    print("\n--- Good Prediction ---")
    result_good = table_first_matching_reward(prediction_good, reference)
    for key, value in result_good.items():
        print(f"  {key}: {value}")
    
    print("\n--- Bad Prediction (Table→Text, Text→Table errors) ---")
    result_bad = table_first_matching_reward(prediction_bad, reference)
    for key, value in result_bad.items():
        print(f"  {key}: {value}")
    
    print("\n--- Comparison ---")
    print(f"Good prediction normalized reward: {result_good['normalized']:.3f}")
    print(f"Bad prediction normalized reward: {result_bad['normalized']:.3f}")


# =============================================================================
# TEDS-BASED TABLE STRUCTURE REWARD
# =============================================================================

def extract_tables_from_text(text: str) -> List[str]:
    """
    Extract all <table>...</table> HTML blocks from text.
    Excludes <table_block> wrapper tags.
    
    Args:
        text: Text containing table blocks
        
    Returns:
        List of table HTML strings
    """
    # Pattern to match <table>...</table> (not <table_block>)
    # Use negative lookahead to exclude _block
    pattern = r'<table(?!_block)[^>]*>.*?</table>'
    tables = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return tables


def compute_teds_reward(
    pred_tables: List[str],
    ref_tables: List[str],
    structure_only: bool = False
) -> Dict[str, float]:
    """
    Compute TEDS-based reward for table matching.
    Uses Hungarian algorithm to find optimal table pairing.
    
    Args:
        pred_tables: List of predicted table HTML strings
        ref_tables: List of reference table HTML strings
        structure_only: If True, use TEDS-S (structure only)
        
    Returns:
        Dict with scores and details
    """
    from teds import compute_teds
    
    n_pred = len(pred_tables)
    n_ref = len(ref_tables)
    
    # Edge cases
    if n_ref == 0 and n_pred == 0:
        return {
            'mean_teds': 1.0,
            'matched_pairs': 0,
            'extra_pred': 0,
            'missing_ref': 0,
            'penalty': 0.0,
        }
    
    if n_ref == 0:
        # Predicted tables when there shouldn't be any
        return {
            'mean_teds': 0.0,
            'matched_pairs': 0,
            'extra_pred': n_pred,
            'missing_ref': 0,
            'penalty': -0.5 * n_pred,  # Penalty for false positives
        }
    
    if n_pred == 0:
        # Missing tables
        return {
            'mean_teds': 0.0,
            'matched_pairs': 0,
            'extra_pred': 0,
            'missing_ref': n_ref,
            'penalty': -1.0 * n_ref,  # Higher penalty for missing tables
        }
    
    # Compute TEDS score matrix for all pairs
    score_matrix = np.zeros((n_pred, n_ref))
    for i, pred_table in enumerate(pred_tables):
        for j, ref_table in enumerate(ref_tables):
            score_matrix[i, j] = compute_teds(pred_table, ref_table, structure_only=structure_only)
    
    # Use Hungarian algorithm for optimal matching (maximize score = minimize -score)
    cost_matrix = -score_matrix  # Convert to minimization problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Get matched scores
    matched_scores = [score_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    mean_teds = np.mean(matched_scores) if matched_scores else 0.0
    
    # Count unmatched
    n_matched = len(row_ind)
    extra_pred = max(0, n_pred - n_matched)
    missing_ref = max(0, n_ref - n_matched)
    
    # Penalty for unmatched tables
    penalty = -0.5 * extra_pred - 1.0 * missing_ref
    
    return {
        'mean_teds': mean_teds,
        'matched_pairs': n_matched,
        'extra_pred': extra_pred,
        'missing_ref': missing_ref,
        'penalty': penalty,
        'matched_scores': matched_scores,
    }


def table_structure_reward(
    completions: List[Any],
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    GRPO-compatible reward function for table structure using TEDS.
    
    Key principle: COUNT ACCURACY IS MANDATORY FOR POSITIVE REWARD
    
    Scoring:
    - Perfect count + perfect quality = +2.0
    - Perfect count + imperfect quality = positive (scaled by quality)
    - Wrong count = NEGATIVE (even if matched tables are perfect)
    
    Args:
        completions: List of model completions
        answer: List of reference answers
        
    Returns:
        List of reward scores in range [-2, 2]
    """
    rewards = []
    
    for completion, reference in zip(completions, answer):
        completion_text = extract_text_from_completion(completion)
        reference_text = extract_text_from_completion(reference)
        
        pred_tables = extract_tables_from_text(completion_text)
        ref_tables = extract_tables_from_text(reference_text)
        
        n_pred = len(pred_tables)
        n_ref = len(ref_tables)
        
        # Case 1: Both have no tables - perfect
        if n_pred == 0 and n_ref == 0:
            rewards.append(1.5)
            continue
        
        # Case 2: Reference has tables but prediction has none
        if n_pred == 0 and n_ref > 0:
            rewards.append(max(-2.0, -1.2 * n_ref))
            continue
        
        # Case 3: Prediction has tables but reference has none (hallucination)
        if n_pred > 0 and n_ref == 0:
            rewards.append(max(-2.0, -0.8 * n_pred))
            continue
        
        # Case 4: Both have tables
        teds_result = compute_teds_reward(pred_tables, ref_tables, structure_only=False)
        teds_s_result = compute_teds_reward(pred_tables, ref_tables, structure_only=True)
        
        n_matched = teds_result['matched_pairs']
        
        if n_matched == 0:
            rewards.append(-1.5)
            continue
        
        # Quality of matched tables (0 to 1)
        quality = 0.5 * teds_result['mean_teds'] + 0.5 * teds_s_result['mean_teds']
        
        # COUNT ACCURACY RATIO:
        # Perfect: n_pred == n_ref and all matched -> ratio = 1.0
        # Extra tables: ratio < 1 (penalize hallucinations)
        # Missing tables: ratio < 1 (penalize omissions)
        
        # Jaccard-like count accuracy: intersection / union
        count_accuracy = n_matched / max(n_pred, n_ref)
        
        # CRITICAL: If count_accuracy < 1, the model made errors
        # We want: count_accuracy=1 -> can be positive
        #          count_accuracy<1 -> should be negative or low
        
        if count_accuracy >= 1.0:
            # Perfect count match - reward based on quality
            # quality=1.0 -> +2.0, quality=0.5 -> +0.5
            final_score = (quality * 3.0) - 1.0
        else:
            # Count mismatch - penalize proportionally
            # Even with perfect quality, wrong count = negative
            
            # Formula: (count_accuracy - 0.6) * 5 * quality
            # This shifts threshold to 0.6, so:
            # count_acc=0.5 (1 extra): (0.5-0.6)*5*1.0 = -0.5
            # count_acc=0.33 (2 extra): (0.33-0.6)*5*1.0 = -1.35
            # count_acc=0.67: (0.67-0.6)*5*1.0 = +0.35
            final_score = (count_accuracy - 0.6) * 5.0 * quality
        
        final_score = max(-2.0, min(2.0, final_score))
        rewards.append(final_score)
    
    return rewards


def table_structure_reward_strict(
    completions: List[Any],
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    Stricter version focusing on TEDS-S (structure only).
    Useful when content extraction is good but structure matters more.
    
    Args:
        completions: List of model completions
        answer: List of reference answers
        
    Returns:
        List of reward scores
    """
    rewards = []
    
    for completion, reference in zip(completions, answer):
        completion_text = extract_text_from_completion(completion)
        reference_text = extract_text_from_completion(reference)
        
        pred_tables = extract_tables_from_text(completion_text)
        ref_tables = extract_tables_from_text(reference_text)
        
        # Only use TEDS-S for strict structure matching
        teds_s_result = compute_teds_reward(pred_tables, ref_tables, structure_only=True)
        
        # Simple scoring: TEDS-S + penalty
        score = teds_s_result['mean_teds'] + teds_s_result['penalty'] * 0.3
        
        # Scale to [-2, 2]
        final_score = (score * 4) - 2
        final_score = max(-2.0, min(2.0, final_score))
        
        rewards.append(final_score)
    
    return rewards


if __name__ == "__main__":
    test_matching()
    
    # Test TEDS reward
    print("\n" + "=" * 60)
    print("Testing TEDS Reward Functions")
    print("=" * 60)
    
    test_pred = """
<table_block>
<table>
<tr><td>Container</td><td>Weight</td></tr>
<tr><td>MSKU123</td><td>15000 KG</td></tr>
</table>
</table_block>
"""
    
    test_ref = """
<table_block>
<table>
<tr><td>Container</td><td>Weight</td></tr>
<tr><td>MSKU123</td><td>15000 KG</td></tr>
</table>
</table_block>
"""
    
    test_ref_diff = """
<table_block>
<table>
<tr><td>Container No.</td><td>Weight (KG)</td></tr>
<tr><td>TCLU456</td><td>18000</td></tr>
</table>
</table_block>
"""
    
    print("\nTest 1: Identical tables")
    rewards = table_structure_reward([test_pred], [test_ref])
    print(f"  Reward: {rewards[0]:.4f}")
    
    print("\nTest 2: Different content, same structure")
    rewards = table_structure_reward([test_pred], [test_ref_diff])
    print(f"  Reward: {rewards[0]:.4f}")
    
    print("\nTest 3: No tables in prediction")
    rewards = table_structure_reward(["<text_block>No tables here</text_block>"], [test_ref])
    print(f"  Reward: {rewards[0]:.4f}")
    
    print("\nTest 4: Extra tables in prediction")
    extra_pred = test_pred + test_pred  # Two tables
    rewards = table_structure_reward([extra_pred], [test_ref])
    print(f"  Reward: {rewards[0]:.4f}")
