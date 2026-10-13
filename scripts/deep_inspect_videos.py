import os
from pathlib import Path
import cv2

def inspect_dir(path_str):
    p = Path(path_str)
    print(f"\n==========================================")
    print(f"Deep Inspection of: {path_str}")
    print(f"==========================================")
    
    classes = {}
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    # Recursive search for video files
    total_videos = 0
    sample_video = None
    
    for root, dirs, files in os.walk(p):
        vids = [f for f in files if Path(f).suffix.lower() in video_extensions]
        if vids:
            rel = os.path.relpath(root, p)
            classes[rel] = len(vids)
            total_videos += len(vids)
            if sample_video is None and vids:
                sample_video = Path(root) / vids[0]
                
    print(f"Total video subdirectories/classes: {len(classes)}")
    print(f"Total video files: {total_videos}")
    print("\nSample classes and counts:")
    for k in sorted(classes.keys())[:25]:
        print(f"  - {k}: {classes[k]} videos")
        
    if sample_video:
        print(f"\nProbing sample video: {sample_video}")
        cap = cv2.VideoCapture(str(sample_video))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dur = frame_count / fps if fps > 0 else 0
            print(f"  Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {frame_count}, Duration: {dur:.2f}s")
            cap.release()
        else:
            print("  Could not open video with OpenCV!")

if __name__ == "__main__":
    inspect_dir(r"C:\Users\navee\Downloads\archive\ProcessedData_vivit")
    inspect_dir(r"C:\Users\navee\Downloads\archive (2)\Video_Dataset")
    inspect_dir(r"C:\Users\navee\Downloads\archive (2)\Sample Videos")
