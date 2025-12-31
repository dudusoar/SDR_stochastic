#!/usr/bin/env python
"""
Fix syntax errors in generators.py by replacing escaped quotes.
"""

import re
import sys

def fix_generators_file(filepath):
    """Fix all escaped quote patterns in generators.py"""
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Track changes
    changes = []
    
    # Fix f\" patterns (should be f")
    # Pattern matches f\\\" (escaped backslash and quote)
    pattern1 = r'f\\\\\\\"'
    if re.search(pattern1, content):
        content = re.sub(pattern1, 'f\"', content)
        changes.append("Fixed f\\\\\\\" patterns")
    
    # Fix f\\\" patterns (double escaped)
    pattern2 = r'f\\\\\\\\\"'
    if re.search(pattern2, content):
        content = re.sub(pattern2, 'f\"', content)
        changes.append("Fixed f\\\\\\\\\" patterns")
    
    # Fix \\\" patterns (escaped quotes in general)
    pattern3 = r'\\\\\\\\\\\"'
    if re.search(pattern3, content):
        content = re.sub(pattern3, '\"', content)
        changes.append("Fixed \\\\\\\" patterns")
    
    # Fix \\\" patterns (double backslash quote)
    pattern4 = r'\\\\\\\\\"'
    if re.search(pattern4, content):
        content = re.sub(pattern4, '\"', content)
        changes.append("Fixed \\\\\" patterns")
    
    # Fix f\" patterns with single backslash (should be f")
    pattern5 = r'f\\\\\"'
    if re.search(pattern5, content):
        content = re.sub(pattern5, 'f\"', content)
        changes.append("Fixed f\\\" patterns")
    
    # Fix \" patterns with single backslash (should be ")
    pattern6 = r'\\\\\"'
    if re.search(pattern6, content):
        content = re.sub(pattern6, '\"', content)
        changes.append("Fixed \\\" patterns")
    
    # Write the file back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return changes

if __name__ == '__main__':
    filepath = 'vrp_toolkit/data/generators.py'
    print(f"Fixing {filepath}...")
    
    try:
        changes = fix_generators_file(filepath)
        
        if changes:
            print("Changes made:")
            for change in changes:
                print(f"  • {change}")
        else:
            print("No patterns found to fix.")
        
        # Test compilation
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', filepath],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ File compiles successfully!")
        else:
            print("✗ Compilation failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)