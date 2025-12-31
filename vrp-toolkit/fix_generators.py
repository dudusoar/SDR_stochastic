import re

# Read the file
with open('vrp_toolkit/data/generators.py', 'r') as f:
    content = f.read()

# Replace f\"...\" with f'...'
# Pattern to match f\"...\"
pattern1 = r'f\\\\\\\"\[.*?\]\\\\\\\"'
replacement1 = "f'[{', '.join(map(str, orders))}]'"
content = re.sub(pattern1, replacement1, content)

pattern2 = r'f\\\\\\\"{orders\[0\]}\\\\\\\"'
replacement2 = "f'{orders[0]}'"
content = re.sub(pattern2, replacement2, content)

# Write back
with open('vrp_toolkit/data/generators.py', 'w') as f:
    f.write(content)

print("File fixed")