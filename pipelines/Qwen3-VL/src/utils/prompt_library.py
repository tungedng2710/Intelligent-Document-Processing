"""
Prompt Library for Qwen3-VL Document Parser
Centralized prompt management system
"""

from pathlib import Path
from typing import Dict, Optional
import os

class PromptLibrary:
    """Manage and load prompts from the prompts directory."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompt library.
        
        Args:
            prompts_dir: Path to prompts directory. If None, uses default location.
        """
        if prompts_dir is None:
            # Default to prompts/ directory at project root
            # Go up two levels from src/utils/ to get to project root
            current_dir = Path(__file__).parent.parent.parent
            self.prompts_dir = current_dir / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        self._cache: Dict[str, str] = {}
        
    def load_prompt(self, name: str, use_cache: bool = True) -> str:
        """
        Load a prompt by name.
        
        Args:
            name: Prompt filename (with or without .txt extension)
            use_cache: Whether to use cached prompts
            
        Returns:
            Prompt text as string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Normalize name
        if not name.endswith('.txt'):
            name = f"{name}.txt"
        
        # Check cache first
        if use_cache and name in self._cache:
            return self._cache[name]
        
        # Load from file
        prompt_path = self.prompts_dir / name
        
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt '{name}' not found at {prompt_path}\n"
                f"Available prompts: {self.list_prompts()}"
            )
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cache it
        if use_cache:
            self._cache[name] = content
        
        return content
    
    def list_prompts(self) -> list:
        """List all available prompts."""
        if not self.prompts_dir.exists():
            return []
        
        return sorted([
            f.stem for f in self.prompts_dir.glob("*.txt")
        ])
    
    def get_prompt(self, name: str) -> str:
        """Alias for load_prompt with cache enabled."""
        return self.load_prompt(name, use_cache=True)
    
    def reload(self, name: Optional[str] = None):
        """
        Reload prompt(s) from disk, bypassing cache.
        
        Args:
            name: Specific prompt to reload, or None to reload all
        """
        if name is None:
            self._cache.clear()
        else:
            if not name.endswith('.txt'):
                name = f"{name}.txt"
            self._cache.pop(name, None)
    
    def add_prompt(self, name: str, content: str):
        """
        Add or update a prompt.
        
        Args:
            name: Prompt filename (will add .txt if not present)
            content: Prompt text content
        """
        if not name.endswith('.txt'):
            name = f"{name}.txt"
        
        prompt_path = self.prompts_dir / name
        
        # Create directory if it doesn't exist
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Write prompt
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update cache
        self._cache[name] = content
    
    def __repr__(self):
        return f"PromptLibrary(prompts_dir={self.prompts_dir}, cached={len(self._cache)})"


# Default prompt mappings
DEFAULT_PROMPTS = {
    'api': 'document_extraction_v1',
    'api_semantic': 'document_extraction_v2_semantic',
    'training_base': 'training_base',
    'training_sft': 'training_sft_tagged',
}


def get_default_library() -> PromptLibrary:
    """Get the default prompt library instance."""
    return PromptLibrary()


def get_prompt(name: str, library: Optional[PromptLibrary] = None) -> str:
    """
    Convenience function to get a prompt.
    
    Args:
        name: Prompt name (can use aliases from DEFAULT_PROMPTS)
        library: PromptLibrary instance, or None to use default
        
    Returns:
        Prompt text
        
    Examples:
        >>> prompt = get_prompt('api')
        >>> prompt = get_prompt('document_extraction_v1')
    """
    if library is None:
        library = get_default_library()
    
    # Check if name is an alias
    if name in DEFAULT_PROMPTS:
        name = DEFAULT_PROMPTS[name]
    
    return library.get_prompt(name)
