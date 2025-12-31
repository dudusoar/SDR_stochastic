#!/usr/bin/env python
"""
Fix all escaped quote patterns in generators.py
"""

import re

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Debug: show problematic line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'index=[' in line and 'f' in line:
            print(f"Line {i+1}: {repr(line)}")
    
    # Fix patterns systematically
    # Replace f\\\" with f"
    original = content
    content = re.sub(r'f\\\\\\\"', 'f\"', content)
    
    # Replace \\\" with "
    content = re.sub(r'\\\\\\\\\\\"', '\"', content)
    
    # Replace f\\\" with f" (fewer backslashes)
    content = re.sub(r'f\\\\\"', 'f\"', content)
    
    # Replace \\\" with " (fewer backslashes)
    content = re.sub(r'\\\\\"', '\"', content)
    
    if original != content:
        print("File modified")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    else:
        print("No changes made")
        return False

if __name__ == '__main__':
    fix_file('vrp_toolkit/data/generators.py')
    
    # Test compilation
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'py_compile', 'vrp_toolkit/data/generators.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("SUCCESS: File compiles")
    else:
        print("FAILED:")
        print(result.stderr)