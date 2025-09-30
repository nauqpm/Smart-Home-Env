# train_bc.py
"""
Train Behavior Cloning DNNs for shiftable loads from expert dataset.
Usage: python train_bc.py output/example_expert_schedule.json
Dependencies: torch, numpy, scikit-learn
"""
import json
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
except Exception:
    print("PyTorch or sklearn not installed. This script file is created; run it where these libs are installed.")
    # still allow user to view code

class ShiftableDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BCNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def build_dataset_from_schedule(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    price = np.array(data['price'])
    pv = np.array(data['pv'])
    schedule = data['schedule']
    z_su = np.array(schedule['z_su'])  # N_su x T
    T = len(price)
    datasets = []
    N_su = z_su.shape[0] if z_su.size else 0
    for i in range(N_su):
        X_i = []
        Y_i = []
        for t in range(T):
            X_i.append([t / (T-1), price[t], pv[t]])
            Y_i.append(int(z_su[i,t]))
        datasets.append((np.array(X_i), np.array(Y_i)))
    return datasets

def train_single(dset, epochs=80, batch_size=32, wpos=2.5):
    X, y = dset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    ds_train = ShiftableDataset(X_train, y_train)
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    model = BCNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss(reduction='none')
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            pred = model(xb)
            loss_raw = bce(pred, yb)
            weights = torch.where(yb==1, torch.tensor(wpos, dtype=torch.float32), torch.tensor(1.0))
            loss = (loss_raw * weights).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if ep % 10 == 0:
            print(f"ep {ep}, loss {np.mean(losses):.4f}")
    # test
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        preds = (model(X_test_t).numpy() > 0.5).astype(int)
        acc = (preds == y_test).mean()
    print("Test acc", acc)
    return model

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv)>1 else 'output/example_expert_schedule.json'
    datasets = build_dataset_from_schedule(input_path)
    models = []
    os.makedirs("models", exist_ok=True)
    for i, d in enumerate(datasets):
        print("Train SU", i)
        model = train_single(d)
        models.append(model)
        torch.save(model.state_dict(), f"models/su_agent_{i}.pth")
    print("Saved models in models/")
