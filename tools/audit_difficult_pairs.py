import json

def analyze():
    with open('reports/all_136_class_audit.json') as f:
        audit = json.load(f)

    rows = audit['rows']
    target_pairs = [
        ('clothing', 'dress'), ('dress', 'loose'), ('narrow', 'shirt'),
        ('night', 'hat'), ('small', 'short'), ('tea', 'giraffe'),
        ('yesterday', 'hat'), ('young', 'fast')
    ]

    print('=== Target Difficult Pairs Analysis ===')
    row_dict = {r['class']: r for r in rows}
    for c1, c2 in target_pairs:
        r1 = row_dict.get(c1, {})
        r2 = row_dict.get(c2, {})
        c1_to_c2 = [m for m in r1.get('test_confusions', []) if m['predicted_class'] == c2]
        c2_to_c1 = [m for m in r2.get('test_confusions', []) if m['predicted_class'] == c1]
        f1_1 = r1.get('test_f1', 0.0)
        f1_2 = r2.get('test_f1', 0.0)
        print(f"Pair {c1} <-> {c2}:")
        print(f"  {c1} (train={r1.get('train_count')}, val={r1.get('val_count')}, test={r1.get('test_count')}, F1={f1_1:.2f}): confusions -> {c1_to_c2}")
        print(f"  {c2} (train={r2.get('train_count')}, val={r2.get('val_count')}, test={r2.get('test_count')}, F1={f1_2:.2f}): confusions -> {c2_to_c1}")

    # Also list all top confusions in test set
    print("\n=== Top Confusion Pairs Across All 136 Classes ===")
    all_confusions = []
    for r in rows:
        c_gt = r['class']
        for m in r.get('test_confusions', []):
            all_confusions.append((c_gt, m['predicted_class'], m['count']))
    all_confusions.sort(key=lambda x: -x[2])
    for c_gt, c_pred, cnt in all_confusions[:25]:
        print(f"  GT: {c_gt:<15} -> Predicted: {c_pred:<15} (count: {cnt})")

if __name__ == '__main__':
    analyze()
