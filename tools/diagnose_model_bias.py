import torch
import json
import numpy as np

def main():
    ckpt = torch.load('checkpoints/achwgate_136class_production_best.pth', map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    bias = sd['classifier.bias'].numpy()
    
    with open('configs/authoritative_136_classes_manifest.json') as f:
        classes = json.load(f)['classes']
        
    top10_bias = np.argsort(bias)[-10:][::-1]
    bottom10_bias = np.argsort(bias)[:10]
    
    print("Top 10 highest bias classes in model classifier.bias:")
    for i in top10_bias:
        print(f"  {classes[i]} (idx {i}): bias = {bias[i]:.4f}")
        
    print("\nBottom 10 lowest bias classes in model classifier.bias:")
    for i in bottom10_bias:
        print(f"  {classes[i]} (idx {i}): bias = {bias[i]:.4f}")
        
    gm_idx = classes.index('good_morning')
    ty_idx = classes.index('thank_you')
    print(f"\n'good_morning' (idx {gm_idx}) bias: {bias[gm_idx]:.4f}")
    print(f"'thank_you' (idx {ty_idx}) bias: {bias[ty_idx]:.4f}")
    
    # Also inspect weights norm
    weight = sd['classifier.weight'].numpy() # (136, 128)
    weight_norms = np.linalg.norm(weight, axis=1)
    top10_wnorm = np.argsort(weight_norms)[-10:][::-1]
    print("\nTop 10 highest weight norm classes:")
    for i in top10_wnorm:
        print(f"  {classes[i]} (idx {i}): norm = {weight_norms[i]:.4f}")

if __name__ == "__main__":
    main()
