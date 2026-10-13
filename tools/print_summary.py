import json

data = json.load(open('reports/preextracted_landmarks_summary.json'))
print('Total classes:', data['total_classes'])
print('>=50 samples:', data['classes_ge_50'])
print('>=20 samples:', data['classes_ge_20'])
print('>=15 samples:', data['classes_ge_15'])
for i, c in enumerate(data['classes']):
    print(f"{i+1:3d}. {c['class']:20s} : {c['count']:4d}")
