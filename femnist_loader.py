# === imports (top of file) ===
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import flwr as fl
from datasets import load_dataset      # FEMNIST
# --------------------

# === STEP 5: Model definition (put near top) ===
class FEMNISTNet(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # adjust if input size differs
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === STEP 4: Label flip wrapper (introduce when preparing per-client datasets) ===
class LabelFlipDataset(Dataset):
    def __init__(self, base_dataset, malicious=False, flip_map=None):
        """
        base_dataset: an HF Dataset or torch Dataset that returns (image, label)
        malicious: set True to flip labels when __getitem__ is called
        flip_map: dict mapping original_label -> flipped_label
        """
        self.base = base_dataset
        self.malicious = malicious
        self.flip_map = flip_map or {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[int(idx)]
        # HF datasets return dicts; adapt depending on type
        if isinstance(item, dict):
            img = item["image"]        # PIL or numpy array
            label = int(item["label"])
        else:
            img, label = item
        if self.malicious:
            label = self.flip_map.get(label, label)
        # convert to torch tensors if needed (assuming image is already numpy/PIL)
        # transform outside this wrapper normally
        return img, torch.tensor(label, dtype=torch.long)

# === STEP 6: Flower NumPy client (place before launching simulation) ===
class FEMNISTClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_dataset, test_dataset, device="cpu"):
        self.cid = cid
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(1):
            for x, y in loader:
                x = x.float().unsqueeze(1) if x.dim()==3 else x.float()  # ensure [B,1,H,W]
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
        return self.get_parameters(), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=128)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.float().unsqueeze(1) if x.dim()==3 else x.float()
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return 0.0, total, {"accuracy": correct/total}

# === STEP 7: Custom strategy with distance-based filter ===
class DistanceFilterFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # if any failures, fallback:
        if failures:
            return super().aggregate_fit(rnd, results, failures)

        if not results:
            return None, {}

        # convert each client's returned parameters to numpy arrays
        ndarrays_per_client = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
        # flatten each client's params into a vector
        vectors = [np.concatenate([p.flatten() for p in client_params]) for client_params in ndarrays_per_client]
        V = np.vstack(vectors)
        center = np.median(V, axis=0)
        dists = np.linalg.norm(V - center, axis=1)
        med = np.median(dists); q75, q25 = np.percentile(dists, [75,25])
        threshold = med + 1.5 * (q75 - q25)
        keep_mask = dists <= threshold
        filtered = [r for (r, keep) in zip(results, keep_mask) if keep]
        # if everything filtered, fallback to all
        if len(filtered) == 0:
            filtered = results
        print(f"[Round {rnd}] flagged {len(results) - len(filtered)} of {len(results)}")
        return super().aggregate_fit(rnd, filtered, failures=[])

# === main(): where to glue everything together ===
def main():
    # 1) Load FEMNIST
    dataset = load_dataset("flwrlabs/femnist")
    # dataset['train'] has fields: 'image', 'label', 'user_id'
    train = dataset["train"]
    test = dataset["test"]

    # 2) Partition: gather the list of unique user_ids and create per-user splits
    users = list(set(train["user_id"]))
    users.sort()
    num_clients = min(50, len(users))  # decide how many clients to simulate
    selected_users = users[:num_clients]

    # Build per-client HF sub-datasets (you can also convert to torch datasets)
    client_datasets = []
    for u in selected_users:
        subset = train.filter(lambda ex, uid=u: ex["user_id"] == uid)  # small lambda closure
        client_datasets.append(subset)  # HF Dataset

    # 3) Choose malicious clients
    rand = random.Random(42)
    num_malicious = max(1, int(0.2 * num_clients))
    malicious_client_ids = set(rand.sample([str(i) for i in range(num_clients)], num_malicious))
    print("malicious ids:", malicious_client_ids)

    # 4) Prepare flip map (example: rotate labels by 1)
    flip_map = {i: (i + 1) % 62 for i in range(62)}

    # 5) Build client factory for Flower simulation
    def client_fn(cid: str):
        idx = int(cid)
        base_ds = client_datasets[idx]
        # wrap base dataset in LabelFlipDataset if malicious
        is_malicious = cid in malicious_client_ids
        train_ds = LabelFlipDataset(base_ds, malicious=is_malicious, flip_map=flip_map)
        test_ds = LabelFlipDataset(test, malicious=False)  # server-side global testset
        model = FEMNISTNet()
        return FEMNISTClient(cid=cid, model=model, train_dataset=train_ds, test_dataset=test_ds, device="cpu")

    # 6) Strategy and start simulation
    strategy = DistanceFilterFedAvg(fraction_fit=1.0, min_fit_clients=num_clients, min_available_clients=num_clients)
    fl.simulation.start_simulation(client_fn=client_fn, num_clients=num_clients, config=fl.server.ServerConfig(num_rounds=10), strategy=strategy)

if __name__ == "__main__":
    main()
