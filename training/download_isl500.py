"""
Authoritative ISL500 Dataset Downloader for AC-HWGATE.
Downloads temporal MediaPipe landmark sequences for all 500 ISL word classes from:
https://huggingface.co/datasets/ISL500/ISL-DATA
"""
import os
import sys
import re
import json
import time
import requests
import requests.adapters
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dns_patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(BASE_DIR, "isl500_meta")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_isl500")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(META_DIR, "MappingFiles", "PersonDependentTrain.csv")
TEST_CSV = os.path.join(META_DIR, "MappingFiles", "PersonDependentTest.csv")

def parse_row(path: str):
    if not isinstance(path, str) or not path.endswith('.h5'):
        return None
    fn = path.split('/')[-1]
    
    # Extract User
    m_user = re.search(r'(ISL_DATA_USER\d+|user\d+)', path, re.I)
    user = m_user.group(1).upper() if m_user else 'ISL_DATA_USER001'
    if not user.startswith('ISL_DATA_'):
        user = 'ISL_DATA_' + user
        
    # Extract Word
    clean_name = re.sub(r'^.*_([A-Za-z0-9]+)__session', r'\1__session', fn)
    m_word = re.match(r'^([A-Za-z0-9]+)__session', clean_name)
    word = m_word.group(1) if m_word else fn.split('__')[0]
    
    # Strip any prefix artifact
    if '_' in fn and not fn.startswith(word):
        # file on HF is typically just <word>__session... or the full name
        # We will attempt primary name then full name
        primary_fn = re.sub(r'^.*?([A-Za-z0-9]+__session\d+__clip\d+\.h5)$', r'\1', fn)
    else:
        primary_fn = fn
        
    return {
        "user": user,
        "word": word,
        "filename": primary_fn,
        "raw_filename": fn,
        "raw_path": path
    }

def download_file(item, session, max_retries=3):
    user = item['user']
    fn = item['filename']
    word = item['word']
    split = item['split']
    
    target_dir = os.path.join(OUTPUT_DIR, split, word)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, fn)
    
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
        return True, item, target_path
        
    urls_to_try = [
        f"https://huggingface.co/datasets/ISL500/ISL-DATA/resolve/main/Landmarks/MediaPipe/{user}/{fn}",
        f"https://huggingface.co/datasets/ISL500/ISL-DATA/resolve/main/Landmarks/MediaPipe/{user}/{item['raw_filename']}"
    ]
    
    for attempt in range(max_retries):
        for url in urls_to_try:
            try:
                resp = session.get(url, timeout=12)
                if resp.status_code == 200 and len(resp.content) > 1024:
                    with open(target_path, 'wb') as f:
                        f.write(resp.content)
                    return True, item, target_path
            except Exception:
                time.sleep(0.5)
        time.sleep(1.0)
        
    return False, item, None

def main(samples_per_class_train=2, samples_per_class_test=1, max_workers=12):
    print("=" * 70, flush=True)
    print("ISL500 DATASET DOWNLOADER FOR AC-HWGATE", flush=True)
    print(f"Target: 500 ISL Word Classes ({samples_per_class_train} train + {samples_per_class_test} test per class)", flush=True)
    print("=" * 70, flush=True)
    
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    
    train_by_class = defaultdict(list)
    for p in df_tr.iloc[:, 0]:
        item = parse_row(p)
        if item:
            item['split'] = 'train'
            train_by_class[item['word']].append(item)
            
    test_by_class = defaultdict(list)
    for p in df_te.iloc[:, 0]:
        item = parse_row(p)
        if item:
            item['split'] = 'test'
            test_by_class[item['word']].append(item)
            
    all_words = sorted(list(train_by_class.keys()))
    print(f"[+] Total verified ISL word classes: {len(all_words)}", flush=True)
    
    # Select download tasks ensuring diverse signers/users per class
    tasks = []
    for word in all_words:
        # Train samples
        tr_candidates = train_by_class.get(word, [])
        # Prefer distinct users
        seen_users = set()
        tr_selected = []
        for c in tr_candidates:
            if c['user'] not in seen_users:
                tr_selected.append(c)
                seen_users.add(c['user'])
                if len(tr_selected) >= samples_per_class_train:
                    break
        # Fallback if not enough distinct users
        if len(tr_selected) < samples_per_class_train:
            for c in tr_candidates:
                if c not in tr_selected:
                    tr_selected.append(c)
                    if len(tr_selected) >= samples_per_class_train:
                        break
        tasks.extend(tr_selected)
        
        # Test samples
        te_candidates = test_by_class.get(word, [])
        te_selected = te_candidates[:samples_per_class_test]
        if not te_selected and len(tr_candidates) > samples_per_class_train:
            # Fallback to extra train sample for test
            fallback = tr_candidates[samples_per_class_train]
            fallback['split'] = 'test'
            te_selected = [fallback]
        tasks.extend(te_selected)
        
    print(f"[+] Total download tasks planned: {len(tasks)} files across {len(all_words)} classes", flush=True)
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers*2)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    
    successful = 0
    failed = 0
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_file, item, session): item for item in tasks}
        
        for i, fut in enumerate(as_completed(futures)):
            success, item, path = fut.result()
            if success:
                successful += 1
            else:
                failed += 1
                
            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"[{i+1}/{len(tasks)}] Progress: {successful} downloaded, {failed} failed | {rate:.1f} files/sec", flush=True)
                
    print("=" * 70, flush=True)
    print(f"[+] Download complete: {successful} files saved to {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
