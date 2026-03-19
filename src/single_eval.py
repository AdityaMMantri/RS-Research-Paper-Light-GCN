import torch
import numpy as np
import scipy.sparse as sp
import math
from collections import defaultdict
from tqdm import tqdm
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
def load_data(path):
    data = defaultdict(set)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for i in parts[1:]:
                data[u].add(int(i))
    return data

train_data = load_data(r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\data\gowalla\train.txt")
test_data  = load_data(r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\data\gowalla\test.txt")

num_users = max(max(train_data.keys()), max(test_data.keys())) + 1
num_items = max(max([max(v) for v in train_data.values()]),
                max([max(v) for v in test_data.values()])) + 1

print(f"Users: {num_users}, Items: {num_items}")

# =========================
# BUILD ADJ MATRIX (SYM)
# =========================
def build_norm_adj(train_dict):
    user_deg = np.zeros(num_users)
    item_deg = np.zeros(num_items)

    edge_u, edge_i = [], []

    for u, items in train_dict.items():
        user_deg[u] = len(items)
        for i in items:
            item_deg[i] += 1
            edge_u.append(u)
            edge_i.append(i)

    rows, cols, vals = [], [], []

    for u, i in zip(edge_u, edge_i):
        du = user_deg[u]
        di = item_deg[i]

        if du == 0 or di == 0:
            continue

        ui = num_users + i
        w = 1.0 / (math.sqrt(du) * math.sqrt(di))

        rows += [u, ui]
        cols += [ui, u]
        vals += [w, w]

    A = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(num_users + num_items, num_users + num_items)
    )

    indices = torch.from_numpy(np.vstack((A.row, A.col))).long()
    values  = torch.from_numpy(A.data).float()

    return torch.sparse_coo_tensor(indices, values, A.shape).coalesce().to(device)

A_norm = build_norm_adj(train_data)

# =========================
# MODEL (SINGLE VERSION)
# =========================
class LightGCNSingle(torch.nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers, A_norm):
        super().__init__()
        self.A_norm = A_norm
        self.n_layers = n_layers
        self.n_users = n_users

        self.user_emb = torch.nn.Embedding(n_users, dim)
        self.item_emb = torch.nn.Embedding(n_items, dim)

    def propagate(self):
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        # ❗ ONLY LAST LAYER (NO MEAN)
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.A_norm, E)

        return E[:self.n_users], E[self.n_users:]

    def get_embeddings(self):
        return self.propagate()

# =========================
# EVALUATION
# =========================
@torch.no_grad()
def evaluate(model, k=20):
    model.eval()
    u_emb, i_emb = model.get_embeddings()
    i_emb_T = i_emb.T

    recall, ndcg = 0.0, 0.0
    users = [u for u in test_data if len(test_data[u]) > 0]

    for u in tqdm(users, desc="Evaluating"):
        scores = torch.matmul(u_emb[u], i_emb_T).cpu().numpy()

        # remove train items
        for ti in train_data.get(u, []):
            scores[ti] = -np.inf

        top_k = np.argpartition(-scores, k)[:k]
        top_k = top_k[np.argsort(-scores[top_k])]

        gt = test_data[u]

        hits = len(set(top_k) & gt)
        recall += hits / min(len(gt), k)

        dcg = sum(1/np.log2(i+2) for i,x in enumerate(top_k) if x in gt)
        idcg = sum(1/np.log2(i+2) for i in range(min(len(gt), k)))

        ndcg += dcg/idcg if idcg > 0 else 0

    n = len(users)
    return recall/n, ndcg/n

# =========================
# RUN ALL MODELS
# =========================
base_path = r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\results\tables\LIGHT-GCN-SINGLE"

models = {
    "k1": "light-gcn_single_k1",
    "k2": "light-gcn_single_k2",
    "k3": "light-gcn_single_k3",
    "k4": "light-gcn_single_k4",
}

results = {}

for k, folder in models.items():
    print(f"\n================ {k} =================")

    model_path = os.path.join(base_path, folder, "best_model.pt")

    checkpoint = torch.load(model_path, map_location=device)

    # 🔥 FIX: infer layers from k
    layers = int(k[1])

    model = LightGCNSingle(num_users, num_items, 64, layers, A_norm).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    r, n = evaluate(model)

    results[k] = {
        "recall@20": float(r),
        "ndcg@20": float(n)
    }

    print(f"{k} → Recall@20: {r:.4f}, NDCG@20: {n:.4f}")

# =========================
# SAVE RESULTS
# =========================
with open("single_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n===== FINAL RESULTS =====")
for k,v in results.items():
    print(k, v)