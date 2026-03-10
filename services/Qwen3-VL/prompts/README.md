# Prompts Library

Centralized prompt management for the Qwen3-VL document parser.

## Structure

This directory contains all prompts used across different components of the system:

```
prompts/
├── README.md                           # This file
├── document_extraction_v1.txt          # API prompt with block tagging
├── document_extraction_v2_semantic.txt # Semantic grouping version
├── training_base.txt                   # Base training prompt
└── training_sft_tagged.txt             # SFT with tagged blocks
```

## Prompt Versions

### API Prompts

#### `document_extraction_v1.txt` (Default API)
- **Used by**: `api_vllm.py`, `api_vllm_2.py`
- **Features**:
  - Block classification (`<text_block>` vs `<table_block>`)
  - HTML table format
  - Checkbox notation
  - HTML entity support
- **Best for**: Production API with structured output

#### `document_extraction_v2_semantic.txt`
- **Used by**: `app.py` (Gradio interface)
- **Features**:
  - Semantic block grouping
  - Visual boundary detection
  - Separator-based output (`---`)
  - More flexible format
- **Best for**: Interactive testing and exploration

### Training Prompts

#### `training_base.txt`
- **Used by**: `unsloth_finetuning.py`
- **Features**: Basic semantic extraction
- **Best for**: Initial training baseline

#### `training_sft_tagged.txt`
- **Used by**: `unsloth_finetuning_sft_tagged.py`
- **Features**: Block tagging with structured output
- **Best for**: SFT training with tagged data

## Usage

### In Python Code

```python
from utils.prompt_library import get_prompt

# Load a prompt by name
prompt = get_prompt('api')

# Or use alias
prompt = get_prompt('document_extraction_v1')
```

### Available Aliases

```python
DEFAULT_PROMPTS = {
    'api': 'document_extraction_v1',
    'api_semantic': 'document_extraction_v2_semantic',
    'training_base': 'training_base',
    'training_sft': 'training_sft_tagged',
}
```

### Advanced Usage

```python
from utils.prompt_library import PromptLibrary

# Create library instance
library = PromptLibrary()

# List available prompts
prompts = library.list_prompts()
print(prompts)  # ['document_extraction_v1', 'document_extraction_v2_semantic', ...]

# Load specific prompt
content = library.load_prompt('document_extraction_v1')

# Reload from disk (bypass cache)
library.reload('document_extraction_v1')

# Add new prompt
library.add_prompt('custom_prompt', "My custom prompt text...")
```

## Creating New Prompts

### 1. Manual Creation

Create a new `.txt` file in this directory:

```bash
cd prompts
nano my_custom_prompt.txt
```

### 2. Programmatic Creation

```python
from utils.prompt_library import PromptLibrary

lib = PromptLibrary()
lib.add_prompt('my_custom_prompt', """
**Task:** Your task description here...

**Format:**
- Your format rules...

**Output:**
""")
```

### 3. Copy Existing

```bash
cp document_extraction_v1.txt my_variation.txt
# Edit as needed
```

## Prompt Design Guidelines

### Structure

All prompts should follow this structure:

```markdown
**Task:** Clear description of what to do

**Format:** or **Grouping Logic:**
- Detailed rules
- Formatting instructions
- Special cases

**Output:**
(leave empty - model fills this)
```

### Best Practices

1. **Be Specific**: Clear, unambiguous instructions
2. **Include Examples**: Show desired output format
3. **Handle Edge Cases**: Address checkboxes, special symbols, etc.
4. **Consistent Formatting**: Use markdown formatting
5. **Test Thoroughly**: Validate on diverse documents

### Common Elements

- **Block Types**: text_block, table_block
- **Formatting**: HTML tables, markdown headers
- **Special Characters**: HTML entities (&copy;, &reg;, etc.)
- **Separators**: `---`, `\n\n`, block tags
- **Headers**: `##` for section titles

## Prompt Selection Guide

| Use Case | Recommended Prompt | Reason |
|----------|-------------------|---------|
| Production API | `document_extraction_v1` | Structured, reliable |
| Interactive Testing | `document_extraction_v2_semantic` | Flexible, readable |
| Training (SFT) | `training_sft_tagged` | Matches training data |
| Training (Base) | `training_base` | Simple baseline |
| Custom Workflow | Create new | Tailor to needs |

## Versioning

When updating prompts:

1. **Minor Changes**: Edit in place, increment version comment
2. **Major Changes**: Create new version (e.g., `v1` → `v2`)
3. **Breaking Changes**: Create new file, update aliases
4. **Test**: Always test before deploying

### Version History

- **v1** (document_extraction_v1): Initial structured format
- **v2** (document_extraction_v2_semantic): Added semantic grouping
- **training_sft_tagged**: Aligned with SFT training data
- **training_base**: Simplified baseline version

## Integration Points

### API Servers
- `src/api/api_vllm.py` → `document_extraction_v1`
- `src/api/api_vllm_2.py` → `document_extraction_v1`
- `src/api/app.py` → `document_extraction_v2_semantic`

### Training Scripts
- `src/training/unsloth_finetuning.py` → `training_base`
- `src/training/unsloth_finetuning_sft_tagged.py` → `training_sft_tagged`

### Inference Scripts
- Can use any prompt via command-line argument or config

## Testing Prompts

### Quick Test

```python
from utils.prompt_library import get_prompt

# Load and print
prompt = get_prompt('api')
print(prompt)
print(f"Length: {len(prompt)} chars")
```

### Integration Test

```bash
# Test in API
cd src/api
python api_vllm_2.py  # Will load from library

# Test in training
cd src/training
python unsloth_finetuning_sft_tagged.py  # Will load from library
```

## Troubleshooting

### Prompt Not Found

```
FileNotFoundError: Prompt 'my_prompt' not found
```

**Solution**: Check filename has `.txt` extension and exists in `prompts/` directory

### Import Error

```
ImportError: cannot import name 'get_prompt'
```

**Solution**: Ensure `src/utils/prompt_library.py` exists and Python path is correct

### Cache Issues

```python
# Force reload from disk
from utils.prompt_library import PromptLibrary
lib = PromptLibrary()
lib.reload()  # Clear all cache
```

## Future Improvements

- [ ] Multi-language prompts
- [ ] Prompt templates with variables
- [ ] A/B testing framework
- [ ] Prompt versioning system
- [ ] Performance metrics per prompt
- [ ] Automatic prompt optimization

## Contributing

When adding new prompts:

1. Follow the structure guidelines
2. Test thoroughly on diverse documents
3. Update this README with description
4. Add to `DEFAULT_PROMPTS` if widely used
5. Document any special use cases
