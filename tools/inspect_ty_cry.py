import os
import glob
import json
import numpy as np

def inspect_classes():
    print("="*70)
    print("INSPECTING DATASET SAMPLES FOR 'thank_you' and 'cry'")
    print("="*70)
    
    ty_files = glob.glob(r"data/additional_landmarks/thank_you/*.npz")
    cry_files = glob.glob(r"data/additional_landmarks/cry/*.npz")
    
    print(f"Total .npz files for thank_you: {len(ty_files)}")
    print(f"Total .npz files for cry: {len(cry_files)}")
    
    if ty_files:
        d = np.load(ty_files[0], allow_pickle=True)
        print("\nThank You sample metadata:")
        print("  Keys:", list(d.keys()))
        print("  Source video:", d.get("source_video", "N/A"))
        print("  Session ID:", d.get("session_id", "N/A"))
        print("  Features shape:", d["features"].shape)
        
    if cry_files:
        d = np.load(cry_files[0], allow_pickle=True)
        print("\nCry sample metadata:")
        print("  Source video:", d.get("source_video", "N/A"))
        print("  Session ID:", d.get("session_id", "N/A"))
        print("  Features shape:", d["features"].shape)
        
    # Check all source videos for thank_you
    ty_sources = set()
    for f in ty_files:
        d = np.load(f, allow_pickle=True)
        ty_sources.add(str(d.get("source_video", "")))
        
    print(f"\nUnique source videos for thank_you ({len(ty_sources)}):")
    for s in sorted(list(ty_sources))[:10]:
        print(" ", s)
        
    # Check all source videos for cry
    cry_sources = set()
    for f in cry_files:
        d = np.load(f, allow_pickle=True)
        cry_sources.add(str(d.get("source_video", "")))
        
    print(f"\nUnique source videos for cry ({len(cry_sources)}):")
    for s in sorted(list(cry_sources))[:10]:
        print(" ", s)

if __name__ == "__main__":
    inspect_classes()
