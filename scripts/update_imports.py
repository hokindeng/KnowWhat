#!/usr/bin/env python3
"""
Script to update imports in Python files to use the core directory structure.
"""

import os
import re
import sys

def update_imports_in_file(filepath):
    """Update imports in a single file to use core directory."""
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Track if we made changes
    original_content = content
    
    # Add sys.path manipulation if needed
    if 'from prompt_builder import' in content or 'from maze_generator import' in content or 'from solution_verifier import' in content or 'from prompts import' in content:
        # Check if sys import already exists
        if 'import sys' not in content:
            # Add after other imports
            import_section = re.search(r'(import.*?\n)+', content)
            if import_section:
                end_pos = import_section.end()
                content = content[:end_pos] + "import sys\n" + content[end_pos:]
        
        # Add path append if not already there
        if 'sys.path.append' not in content:
            # Find where to add it (after imports)
            import_section = re.search(r'(import.*?\n|from.*?\n)+', content)
            if import_section:
                end_pos = import_section.end()
                path_append = "sys.path.append(os.path.join(os.path.dirname(__file__), '../'))\n"
                content = content[:end_pos] + path_append + content[end_pos:]
    
    # Update the imports
    content = re.sub(r'from prompt_builder import', 'from core.prompt_builder import', content)
    content = re.sub(r'from maze_generator import', 'from core.maze_generator import', content)
    content = re.sub(r'from solution_verifier import', 'from core.solution_verifier import', content)
    content = re.sub(r'from prompts import', 'from core.prompts import', content)
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated: {filepath}")
        return True
    return False

def update_directory(directory):
    """Update all Python files in a directory."""
    updated_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            filepath = os.path.join(directory, filename)
            if update_imports_in_file(filepath):
                updated_count += 1
    
    print(f"\nUpdated {updated_count} files in {directory}")

if __name__ == "__main__":
    # Update infer directory
    if os.path.exists('infer'):
        update_directory('infer')
    
    # Update analysis directories
    for subdir in ['visualization', 'statistical', 'core', 'utils']:
        analysis_dir = f'analysis/{subdir}'
        if os.path.exists(analysis_dir):
            update_directory(analysis_dir)
    
    print("\nImport updates complete!") 