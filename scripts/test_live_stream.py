import urllib.request, json, time
import numpy as np

x_test = np.load('data/unified_v2/X_test.npy')
y_test = np.load('data/unified_v2/y_test.npy')
with open('configs/label_map.json') as f:
    lm = json.load(f)
cnames = lm.get('classes', [lm['idx_to_label'][str(i)] for i in range(40)])

for idx in [0, 8, 15, 25, 35]:
    true_label = cnames[y_test[idx]]
    sample = x_test[idx] # (32, 133, 6)
    print(f"\n--- Testing Sample {idx}: True Sign = {true_label} ---")
    
    last_res = None
    for t in range(32):
        kps = sample[t, :, :2].tolist()
        confs = [[0.95] for _ in range(133)]
        payload = {'keypoints': kps, 'confidences': confs}
        req = urllib.request.Request(
            'http://127.0.0.1:8000/predict',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        resp = urllib.request.urlopen(req)
        last_res = json.loads(resp.read().decode('utf-8'))
    
    if last_res is not None:
        print(f"Result: label={last_res['label']}, conf={last_res['confidence']}, status={last_res['status']}, translation=\"{last_res['translation']}\"")
