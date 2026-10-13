import os
from pathlib import Path

paths = [
    r"C:\Users\navee\Downloads\archive\ProcessedData_vivit",
    r"C:\Users\navee\Downloads\archive (2)",
    r"C:\Users\navee\Downloads\archive"
]

print("Scanning local paths...\n")
for p in paths:
    if os.path.exists(p):
        print(f"=== FOUND: {p} ===")
        all_items = list(Path(p).iterdir())
        print(f"Total direct children: {len(all_items)}")
        subdirs = [x for x in all_items if x.is_dir()]
        files = [x for x in all_items if x.is_file()]
        print(f"  Directories: {len(subdirs)}, Files: {len(files)}")
        print("  Sample entries:")
        for item in all_items[:15]:
            t = "DIR" if item.is_dir() else f"FILE ({item.stat().st_size} bytes)"
            print(f"    - {item.name} [{t}]")
            
        # If there are subdirectories, inspect sample subdir
        if subdirs:
            first_sub = subdirs[0]
            sub_children = list(first_sub.iterdir())
            print(f"  Sample sub-directory '{first_sub.name}' has {len(sub_children)} items:")
            for sc in sub_children[:5]:
                st = "DIR" if sc.is_dir() else f"FILE ({sc.stat().st_size} bytes)"
                print(f"      * {sc.name} [{st}]")
        print("\n" + "="*50 + "\n")
    else:
        print(f"=== NOT FOUND: {p} ===\n")
