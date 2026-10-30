import json

data = json.load(open('reports/per_class_precision_recall_report.json'))
sorted_classes = sorted(data.items(), key=lambda x: (x[1]['f1'], x[1]['support']), reverse=True)
print('Total classes:', len(sorted_classes))
f1_90 = [c for c, v in sorted_classes if v['f1'] >= 0.90]
f1_80 = [c for c, v in sorted_classes if v['f1'] >= 0.80]
f1_70 = [c for c, v in sorted_classes if v['f1'] >= 0.70]
print('Classes with F1 >= 90%:', len(f1_90))
print('Classes with F1 >= 80%:', len(f1_80))
print('Classes with F1 >= 70%:', len(f1_70))
print('\n--- ALL CLASSES RANKED ---')
for i, (c, v) in enumerate(sorted_classes):
    print(f"{i+1:02d}. {c:<12}: F1={v['f1']*100:5.1f}%  Prec={v['precision']*100:5.1f}%  Rec={v['recall']*100:5.1f}%  Supp={v['support']}")

