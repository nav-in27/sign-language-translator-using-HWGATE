with open('scripts/build_50class_dataset.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if 'add_files = sorted(glob.glob(str(ADDITIONAL_DIR' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f"{indent}# Exclude 45-degree camera tilts (_left_tilt / _right_tilt) to prevent pose poisoning\n")
        new_lines.append(f"{indent}add_files = [f for f in add_files if not ('_left_tilt' in f or '_right_tilt' in f)]\n")

with open('scripts/build_50class_dataset.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Patch applied successfully.")

