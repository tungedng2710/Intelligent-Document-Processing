#!/usr/bin/env python3
"""
Prompt Management CLI Tool
Manage prompts in the Qwen3-VL prompt library
"""

import argparse
import sys
from pathlib import Path

# Get project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from utils.prompt_library import PromptLibrary, DEFAULT_PROMPTS


def list_prompts(args):
    """List all available prompts."""
    prompts_dir = args.prompts_dir or PROJECT_ROOT / 'prompts'
    lib = PromptLibrary(prompts_dir)
    prompts = lib.list_prompts()
    
    print(f"\n{'='*60}")
    print(f"Available Prompts ({len(prompts)})")
    print(f"{'='*60}\n")
    
    for i, prompt in enumerate(prompts, 1):
        # Check if it has an alias
        aliases = [k for k, v in DEFAULT_PROMPTS.items() if v == prompt]
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        print(f"{i:2d}. {prompt}{alias_str}")
    
    print(f"\n{'='*60}\n")


def show_prompt(args):
    """Show the content of a specific prompt."""
    prompts_dir = args.prompts_dir or PROJECT_ROOT / 'prompts'
    lib = PromptLibrary(prompts_dir)
    
    # Resolve alias if provided
    name = args.name
    if name in DEFAULT_PROMPTS:
        name = DEFAULT_PROMPTS[name]
    
    try:
        content = lib.load_prompt(name)
        
        print(f"\n{'='*60}")
        print(f"Prompt: {name}")
        if args.name in DEFAULT_PROMPTS:
            print(f"Alias: {args.name}")
        print(f"{'='*60}\n")
        print(content)
        print(f"\n{'='*60}")
        print(f"Length: {len(content)} characters")
        print(f"Lines: {content.count(chr(10)) + 1}")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n", file=sys.stderr)
        sys.exit(1)


def create_prompt(args):
    """Create a new prompt."""
    prompts_dir = args.prompts_dir or PROJECT_ROOT / "prompts"
    lib = PromptLibrary(prompts_dir)
    
    # Check if prompt already exists
    try:
        lib.load_prompt(args.name)
        if not args.force:
            print(f"\n❌ Error: Prompt '{args.name}' already exists. Use --force to overwrite.\n", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        pass  # Prompt doesn't exist, good
    
    if args.content:
        content = args.content
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        print("\n📝 Enter prompt content (press Ctrl+D when done):\n")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            content = '\n'.join(lines)
    
    lib.add_prompt(args.name, content)
    print(f"\n✅ Prompt '{args.name}' created successfully!\n")


def delete_prompt(args):
    """Delete a prompt."""
    prompts_dir = args.prompts_dir or PROJECT_ROOT / "prompts"
    lib = PromptLibrary(prompts_dir)
    
    prompt_path = lib.prompts_dir / f"{args.name}.txt"
    
    if not prompt_path.exists():
        print(f"\n❌ Error: Prompt '{args.name}' not found.\n", file=sys.stderr)
        sys.exit(1)
    
    if not args.yes:
        response = input(f"Are you sure you want to delete '{args.name}'? (y/N): ")
        if response.lower() != 'y':
            print("\n❌ Deletion cancelled.\n")
            sys.exit(0)
    
    prompt_path.unlink()
    print(f"\n✅ Prompt '{args.name}' deleted successfully!\n")


def copy_prompt(args):
    """Copy a prompt to a new name."""
    prompts_dir = args.prompts_dir or PROJECT_ROOT / "prompts"
    lib = PromptLibrary(prompts_dir)
    
    try:
        content = lib.load_prompt(args.source)
        
        # Check if destination exists
        try:
            lib.load_prompt(args.dest)
            if not args.force:
                print(f"\n❌ Error: Destination '{args.dest}' already exists. Use --force to overwrite.\n", file=sys.stderr)
                sys.exit(1)
        except FileNotFoundError:
            pass
        
        lib.add_prompt(args.dest, content)
        print(f"\n✅ Prompt copied: '{args.source}' → '{args.dest}'\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n", file=sys.stderr)
        sys.exit(1)


def compare_prompts(args):
    """Compare two prompts."""
    prompts_dir = args.prompts_dir or PROJECT_ROOT / "prompts"
    lib = PromptLibrary(prompts_dir)
    
    try:
        content1 = lib.load_prompt(args.prompt1)
        content2 = lib.load_prompt(args.prompt2)
        
        print(f"\n{'='*60}")
        print(f"Comparing: {args.prompt1} vs {args.prompt2}")
        print(f"{'='*60}\n")
        
        print(f"Prompt 1 ({args.prompt1}):")
        print(f"  Length: {len(content1)} chars")
        print(f"  Lines: {content1.count(chr(10)) + 1}")
        
        print(f"\nPrompt 2 ({args.prompt2}):")
        print(f"  Length: {len(content2)} chars")
        print(f"  Lines: {content2.count(chr(10)) + 1}")
        
        if content1 == content2:
            print(f"\n✅ Prompts are identical\n")
        else:
            print(f"\n⚠️  Prompts are different\n")
            
            # Show first difference
            lines1 = content1.split('\n')
            lines2 = content2.split('\n')
            
            for i, (l1, l2) in enumerate(zip(lines1, lines2), 1):
                if l1 != l2:
                    print(f"First difference at line {i}:")
                    print(f"  {args.prompt1}: {l1[:60]}...")
                    print(f"  {args.prompt2}: {l2[:60]}...")
                    break
        
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Manage prompts in the Qwen3-VL prompt library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all prompts
  %(prog)s list

  # Show a specific prompt
  %(prog)s show api

  # Create new prompt from content
  %(prog)s create my_prompt --content "**Task:** ..."

  # Create from file
  %(prog)s create my_prompt --file prompt.txt

  # Copy existing prompt
  %(prog)s copy api my_api_variation

  # Delete prompt
  %(prog)s delete old_prompt

  # Compare two prompts
  %(prog)s compare api api_semantic
        """
    )
    
    parser.add_argument(
        '--prompts-dir',
        help='Path to prompts directory (default: auto-detect)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    subparsers.add_parser('list', help='List all available prompts')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show prompt content')
    show_parser.add_argument('name', help='Prompt name')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new prompt')
    create_parser.add_argument('name', help='Prompt name')
    create_parser.add_argument('--content', help='Prompt content')
    create_parser.add_argument('--file', help='Read content from file')
    create_parser.add_argument('--force', action='store_true', help='Overwrite if exists')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a prompt')
    delete_parser.add_argument('name', help='Prompt name')
    delete_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    
    # Copy command
    copy_parser = subparsers.add_parser('copy', help='Copy a prompt')
    copy_parser.add_argument('source', help='Source prompt name')
    copy_parser.add_argument('dest', help='Destination prompt name')
    copy_parser.add_argument('--force', action='store_true', help='Overwrite if exists')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two prompts')
    compare_parser.add_argument('prompt1', help='First prompt name')
    compare_parser.add_argument('prompt2', help='Second prompt name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    commands = {
        'list': list_prompts,
        'show': show_prompt,
        'create': create_prompt,
        'delete': delete_prompt,
        'copy': copy_prompt,
        'compare': compare_prompts,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
