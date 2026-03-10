"""
Custom Logits Processor for vLLM v1 API

This processor makes the next token deterministic (argmax) when the last token
matches a target token. It amplifies the max logit significantly so it dominates
regardless of temperature, top_p, top_k, or other sampling settings.
"""

import os
import torch
import logging
from typing import Optional, Any, Dict
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate, MoveDirectionality

# Environment variable to enable logging at all steps (set to "1" or "true" to enable)
LOG_ALL_STEPS = os.environ.get("LOG_ALL_STEPS", "").lower() in ("1", "true", "yes")
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

# Configure logging for the processor
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeterministicAfterTargetProcessor(LogitsProcessor):
    """
    A logits processor that makes token selection deterministic after a target token.
    
    When the last generated token matches the target_token, this processor amplifies
    the maximum logit value so significantly that it becomes the only viable choice,
    effectively making the next token deterministic regardless of sampling parameters.
    
    Usage:
        Pass via extra_args in SamplingParams:
        {
            "extra_args": {
                "target_token": 500,  # Token ID that triggers deterministic selection
                "amplification_factor": 1000.0  # How much to amplify max logit (optional)
            }
        }
    """
    
    # Default amplification factor - makes max logit dominate after softmax
    DEFAULT_AMPLIFICATION = 1000.0
    
    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        """Validate that extra_args contains valid parameters for this processor."""
        if params.extra_args is None:
            return
        
        target_token = params.extra_args.get("target_token")
        if target_token is not None and not isinstance(target_token, int):
            raise ValueError(
                f"target_token must be an integer, got {type(target_token).__name__}: {target_token}"
            )
        
        if target_token is not None and target_token < 0:
            raise ValueError(f"target_token must be non-negative, got {target_token}")
        
        amplification = params.extra_args.get("amplification_factor")
        if amplification is not None:
            if not isinstance(amplification, (int, float)):
                raise ValueError(
                    f"amplification_factor must be a number, got {type(amplification).__name__}"
                )
            if amplification <= 0:
                raise ValueError(f"amplification_factor must be positive, got {amplification}")
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool
    ) -> None:
        """
        Initialize the logits processor.
        
        Args:
            vllm_config: Engine configuration
            device: Hardware accelerator device
            is_pin_memory: Whether pin memory is available
        """
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.vllm_config = vllm_config
        
        # Load tokenizer for decoding token IDs in logs
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        
        # Mapping of batch index -> request info including target_token and output tracking
        self.req_info: Dict[int, Dict[str, Any]] = {}
        
        # Counter to track how many times apply() is called per request
        # This helps debug if output_token_ids is being updated
        self.apply_call_count: Dict[int, int] = {}
    
    def is_argmax_invariant(self) -> bool:
        """
        Return False because this processor can change which token has the highest logit.
        
        When activated (after target token), we amplify the max logit to ensure
        it becomes the selected token deterministically.
        """
        return False
    
    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Update internal state based on batch changes.
        
        Args:
            batch_update: BatchUpdate data structure with add/remove/move operations
        """
        if batch_update is None:
            return
        
        # Debug: Print batch_update structure
        print(f"[LogitsProcessor] batch_update: added={len(batch_update.added)}, "
              f"removed={len(batch_update.removed)}, moved={len(batch_update.moved)}")
        
        # Process added requests first
        # Format: (index, params, prompt_token_ids, output_token_ids)
        # NOTE: output_token_ids is a LIVE REFERENCE that vLLM updates as tokens are generated
        for index, params, prompt_token_ids, output_token_ids in batch_update.added:
            if params is not None:
                self.validate_params(params)
                target_token = None
                amplification = self.DEFAULT_AMPLIFICATION
                if params.extra_args:
                    target_token = params.extra_args.get("target_token")
                    amplification = params.extra_args.get("amplification_factor", self.DEFAULT_AMPLIFICATION)
                
                # Debug: Print detailed info about output_token_ids
                print(f"[LogitsProcessor] Added index={index}: "
                      f"output_token_ids type={type(output_token_ids).__name__}, "
                      f"id={id(output_token_ids)}, "
                      f"len={len(output_token_ids) if output_token_ids else 0}")
                if prompt_token_ids:
                    print(f"[LogitsProcessor] prompt_token_ids type={type(prompt_token_ids).__name__}, "
                          f"len={len(prompt_token_ids) if prompt_token_ids else 0}")
                
                # Store the reference - vLLM should update this list in-place
                self.req_info[index] = {
                    "target_token": target_token,
                    "amplification_factor": amplification,
                    "output_token_ids": output_token_ids,
                    "output_token_ids_id": id(output_token_ids),  # Track object ID for debugging
                }
        
        # Process removed requests
        for index in batch_update.removed:
            self.req_info.pop(index, None)
            self.apply_call_count.pop(index, None)
        
        # Process moved requests - format: (adx, bdx, directionality)
        # Handle both unidirectional moves (a->b) and swaps (a<->b)
        for adx, bdx, direct in batch_update.moved:
            a_val = self.req_info.pop(adx, None)
            b_val = self.req_info.pop(bdx, None)
            if a_val is not None:
                self.req_info[bdx] = a_val
            if direct == MoveDirectionality.SWAP and b_val is not None:
                self.req_info[adx] = b_val
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the logits processor transformation.
        
        For each request where the last token matches target_token:
        1. Find the maximum logit value
        2. Set all logits to -inf except the max
        3. This ensures argmax is selected regardless of sampling params
        
        Args:
            logits: (num_requests, vocab_size) tensor of raw logits
            
        Returns:
            Transformed logits tensor (modified in-place)
        """
        # Helper function to decode token ID to text
        def decode_token(token_id: int) -> str:
            try:
                return repr(self.tokenizer.decode([token_id]))
            except Exception:
                return f"<id:{token_id}>"
        
        # Helper to get last N tokens from a list
        def get_last_n(token_ids, n: int = 3):
            if not token_ids:
                return []
            return list(token_ids[-n:]) if len(token_ids) >= n else list(token_ids)
        
        # Print debug info for all requests in batch
        num_requests = logits.shape[0]
        for batch_idx in range(num_requests):
            # Increment call counter
            self.apply_call_count[batch_idx] = self.apply_call_count.get(batch_idx, 0) + 1
            call_num = self.apply_call_count[batch_idx]
            
            row_logits = logits[batch_idx]
            
            # Get top 3 logits with decoded tokens (these are candidates for next token)
            top_values, top_indices = torch.topk(row_logits, k=min(3, row_logits.shape[0]))
            top3_info = [
                (idx.item(), decode_token(idx.item()), round(val.item(), 2)) 
                for idx, val in zip(top_indices, top_values)
            ]
            
            # Decode top 3 tokens as text for checking
            top3_texts = []
            for idx in top_indices:
                try:
                    top3_texts.append(self.tokenizer.decode([idx.item()]))
                except Exception:
                    top3_texts.append("")
            
            # Check if top1 contains ">" and top2 contains "colspan" or "rowspan" or " "
            # If so, make output deterministic (select top1/argmax)
            top1_text = top3_texts[0] if len(top3_texts) > 0 else ""
            top2_text = top3_texts[1] if len(top3_texts) > 1 else ""
            
            if ">" in top1_text and ("colspan" in top2_text or "rowspan" in top2_text):
                top1_idx = top_indices[0].item()
                top1_logit = top_values[0].item()
                print(f"[LogitsProcessor] Call #{call_num} Batch {batch_idx} | "
                      f"CLOSE_TAG: Forcing top1={repr(top1_text)}({top1_logit:.2f}) over top2={repr(top2_text)} | "
                      f"Top 3: {top3_info}")
                # Set all logits to -inf except top1
                row_logits.fill_(float("-inf"))
                row_logits[top1_idx] = top1_logit
                continue  # Skip further processing for this batch_idx
            
            # Check the output_token_ids from stored reference
            if batch_idx in self.req_info:
                output_ids = self.req_info[batch_idx].get("output_token_ids")
                
                if output_ids is not None and len(output_ids) > 0:
                    last_3 = get_last_n(output_ids, 3)
                    # Decode and concatenate last 3 tokens
                    try:
                        last_3_text = self.tokenizer.decode(last_3)
                    except Exception:
                        last_3_text = ""
                    
                    # Log at all steps if LOG_ALL_STEPS is enabled
                    if LOG_ALL_STEPS:
                        last_3_decoded = [(t, decode_token(t)) for t in last_3]
                        print(f"[LogitsProcessor] Call #{call_num} Batch {batch_idx} | "
                              f"output_len={len(output_ids)} | "
                              f"last_3={last_3_decoded} | "
                              f"Top 3 next: {top3_info}")
                    
                    # Apply logic when last 3 tokens contain "_block>\n"
                    if "_block>\n" in last_3_text:
                        # Get top 2 tokens
                        top2_values, top2_indices = torch.topk(row_logits, k=2)
                        top1_idx = top2_indices[0].item()
                        top2_idx = top2_indices[1].item()
                        top1_logit = top2_values[0].item()
                        top2_logit = top2_values[1].item()
                        
                        # Decode top 1 and top 2 tokens
                        try:
                            top1_text = self.tokenizer.decode([top1_idx])
                            top2_text = self.tokenizer.decode([top2_idx])
                        except Exception:
                            top1_text = ""
                            top2_text = ""
                        
                        # Calculate ratio (handle edge cases)
                        if top2_logit != 0:
                            ratio = top1_logit / top2_logit
                        else:
                            ratio = float('inf') if top1_logit > 0 else 1.0
                        
                        last_3_decoded = [(t, decode_token(t)) for t in last_3]
                        
                        # Check if top1 is "<table" and top2 is "<" and ratio < 1.1
                        if top1_text == "<table" and top2_text == "<":
                            if ratio < 1.1:
                                # Demote top1, amplify top2 so top2 is selected
                                print(f"[LogitsProcessor] Call #{call_num} Batch {batch_idx} | "
                                    f"SWAP: top1={repr(top1_text)}({top1_logit:.2f}) -> top2={repr(top2_text)}({top2_logit:.2f}) | "
                                    f"ratio={ratio:.3f} < 1.1 | last_3={last_3_decoded}")
                                continue
                                # Set all logits to -inf except top2
                                row_logits.fill_(float("-inf"))
                                row_logits[top2_idx] = top2_logit
                        else:
                            # Amplify top1 to make it deterministic
                            print(f"[LogitsProcessor] Call #{call_num} Batch {batch_idx} | "
                                f"KEEP: top1={repr(top1_text)}({top1_logit:.2f}) | "
                                f"ratio={ratio:.3f} | last_3={last_3_decoded}")
                            
                            # Set all logits to -inf except top1
                            row_logits.fill_(float("-inf"))
                            row_logits[top1_idx] = top1_logit
        
        return logits


class DeterministicAfterTargetAdapter(LogitsProcessor):
    """
    Alternative implementation using simpler request-level approach.
    
    This version works with the AdapterLogitsProcessor pattern but implemented
    directly for better control and performance.
    """
    
    DEFAULT_AMPLIFICATION = 1000.0
    
    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        if params.extra_args is None:
            return
        
        target_token = params.extra_args.get("target_token")
        if target_token is not None and not isinstance(target_token, int):
            raise ValueError(f"target_token must be int, got {type(target_token)}")
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool
    ) -> None:
        self.device = device
        self.req_info: Dict[int, Dict[str, Any]] = {}
        # Load tokenizer for decoding token IDs in logs
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
    
    def is_argmax_invariant(self) -> bool:
        return False
    
    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return
        
        # Process added requests - format: (index, params, prompt_token_ids, output_token_ids)
        # NOTE: output_token_ids is a LIVE REFERENCE that vLLM updates
        for index, params, _, output_token_ids in batch_update.added:
            if params is not None:
                target_token = params.extra_args and params.extra_args.get("target_token")
                if target_token is not None:
                    self.req_info[index] = {
                        "target_token": target_token,
                        "output_token_ids": output_token_ids,  # Keep live reference!
                    }
                else:
                    self.req_info.pop(index, None)
        
        if self.req_info:
            # Process removed requests
            for index in batch_update.removed:
                self.req_info.pop(index, None)
            
            # Process moved requests - format: (adx, bdx, directionality)
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_info.pop(adx, None)
                b_val = self.req_info.pop(bdx, None)
                if a_val is not None:
                    self.req_info[bdx] = a_val
                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.req_info[adx] = b_val
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for idx, info in self.req_info.items():
            if idx >= logits.shape[0]:
                continue
            
            output_ids = info.get("output_token_ids")
            target = info["target_token"]
            
            if output_ids and len(output_ids) > 0 and output_ids[-1] == target:
                row = logits[idx]
                max_idx = row.argmax()
                max_val = row[max_idx].clone()
                row.fill_(float("-inf"))
                row[max_idx] = max_val
        
        return logits
