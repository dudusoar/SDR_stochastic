with open('vrp_toolkit/data/generators.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'f\\\\\\\"[' in line:
        line = line.replace('f\\\\\\\"[{', '.join(map(str, orders))}]\\\\\\\"', \"f'[{', '.join(map(str, orders))}]'\")
    if 'f\\\\\\\"{orders[0]}\\\\\\\"' in line:
        line = line.replace('f\\\\\\\"{orders[0]}\\\\\\\"', \"f'{orders[0]}'\")
    new_lines.append(line)

with open('vrp_toolkit/data/generators.py', 'w') as f:
    f.writelines(new_lines)

print('Fixed')