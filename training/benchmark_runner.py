import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import numpy as np

# Import models
from models.cwa_hwgate import CWA_HWGATE
from models.baseline_models import SimpleCNN, SimpleLSTM, SimpleGRU, ST_GCN
from train import ISLAlphabetDataset

def update_status(message, progress):
    with open("benchmark_status.json", "w") as f:
        json.dump({"message": message, "progress": progress}, f)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def train_fast(model, name, train_loader, device):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    epochs = 3 # Fast training for benchmark if weights missing
    for epoch in range(epochs):
        for kps, confs, labels in train_loader:
            kps, confs, labels = kps.to(device), confs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model(kps, confs)
            loss = criterion(log_probs[0], labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), f"{name}_weights.pth")

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for kps, confs, labels in test_loader:
            kps, confs = kps.to(device), confs.to(device)
            log_probs = model(kps, confs)
            preds = torch.argmax(log_probs[0], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    end_time = time.time()
    
    total_time = end_time - start_time
    num_samples = len(test_loader.dataset)
    latency_ms = (total_time / num_samples) * 1000
    fps = 1000 / latency_ms if latency_ms > 0 else 0
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return float(acc), float(f1), float(latency_ms), float(fps)

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = "data_extracted_full"
    
    update_status("Loading dataset...", 5)
    full_dataset = ISLAlphabetDataset(dataset_dir)
    num_classes = len(full_dataset.classes)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    models_to_test = {
        "CNN": SimpleCNN(num_classes=num_classes).to(device),
        "LSTM": SimpleLSTM(num_classes=num_classes).to(device),
        "GRU": SimpleGRU(num_classes=num_classes).to(device),
        "ST-GCN": ST_GCN(num_classes=num_classes).to(device),
        "HWGATE": CWA_HWGATE(num_classes=num_classes).to(device)
    }
    
    results = []
    total_models = len(models_to_test)
    
    for i, (name, model) in enumerate(models_to_test.items()):
        update_status(f"Evaluating {name}...", 10 + int((i / total_models) * 80))
        
        # Check if weights exist
        weights_file = f"{name}_weights.pth"
        if name == "HWGATE":
            weights_file = "cwa_hwgate_pretrained.pth"
            
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file, map_location=device))
        else:
            update_status(f"Training {name} (weights not found)...", 10 + int((i / total_models) * 80) + 5)
            train_fast(model, name, train_loader, device)
            
        acc, f1, latency, fps = evaluate_model(model, test_loader, device)
        size_mb = get_model_size_mb(model)
        
        results.append({
            "model": name,
            "accuracy": acc,
            "f1": f1,
            "latency_ms": latency,
            "fps": fps,
            "model_size_mb": size_mb,
            "is_best": False
        })
        
    # Find best model
    best_idx = max(range(len(results)), key=lambda i: results[i]['accuracy'] + results[i]['f1'])
    results[best_idx]['is_best'] = True
    
    final_output = {
        "dataset_info": {
            "total_samples": len(full_dataset),
            "train_samples": train_size,
            "test_samples": test_size,
            "num_classes": num_classes
        },
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    update_status("Benchmark Complete!", 100)

if __name__ == "__main__":
    run_benchmark()
