import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import math
from tqdm import tqdm
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
def load_data(path):
    data = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            user = parts[0]
            items = parts[1:]
            data[user] = items
    return data

train_data = load_data(r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\data\yelp2018\train.txt")
test_data  = load_data(r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\data\yelp2018\test.txt")

num_users = max(max(train_data.keys()), max(test_data.keys())) + 1

num_items = 0
for u in train_data:
    if len(train_data[u]) > 0:
        num_items = max(num_items, max(train_data[u]))
for u in test_data:
    if len(test_data[u]) > 0:
        num_items = max(num_items, max(test_data[u]))
num_items += 1

print(f"Users: {num_users}, Items: {num_items}")

# =========================
# BUILD GRAPH
# =========================
def build_adj_matrix(train_data, num_users, num_items):
    rows, cols = [], []

    for u in train_data:
        for i in train_data[u]:
            rows.append(u)
            cols.append(i + num_users)
            rows.append(i + num_users)
            cols.append(u)

    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (rows, cols)),
                        shape=(num_users + num_items, num_users + num_items))

    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)

    return (d_mat @ adj @ d_mat).tocsr()

print("Building adjacency matrix...")
norm_adj = build_adj_matrix(train_data, num_users, num_items)

indices = torch.LongTensor(np.vstack((norm_adj.nonzero()[0], norm_adj.nonzero()[1])))
values = torch.FloatTensor(norm_adj.data)
shape = torch.Size(norm_adj.shape)
norm_adj = torch.sparse.FloatTensor(indices, values, shape).to(device)

# =========================
# MODEL
# =========================
class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, embedding_dim)
        self.item_emb = torch.nn.Embedding(num_items, embedding_dim)

    def get_embeddings(self):
        return self.user_emb.weight, self.item_emb.weight

# =========================
# METRICS
# =========================
def recall_at_k(top_k, gt_items):
    return len(set(top_k) & set(gt_items)) / len(gt_items)

def ndcg_at_k(top_k, gt_items):
    dcg = 0
    for i, item in enumerate(top_k):
        if item in gt_items:
            dcg += 1 / math.log2(i + 2)
    idcg = sum([1 / math.log2(i + 2) for i in range(min(len(gt_items), len(top_k)))])
    return dcg / idcg if idcg > 0 else 0

# =========================
# EVALUATE FUNCTION
# =========================
def evaluate_model(model_path, K_layers):
    print(f"\nEvaluating K={K_layers}")

    model = LightGCN(num_users, num_items, 64).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    user_emb, item_emb = model.get_embeddings()
    emb = torch.cat([user_emb, item_emb], dim=0)

    # propagation
    all_emb = [emb]
    for _ in range(K_layers):
        emb = torch.sparse.mm(norm_adj, emb)
        all_emb.append(emb)

    final_emb = torch.mean(torch.stack(all_emb, dim=1), dim=1)

    user_emb = final_emb[:num_users].detach().cpu().numpy()
    item_emb = final_emb[num_users:].detach().cpu().numpy()

    recall_list = []
    ndcg_list = []

    for u in tqdm(test_data, desc=f"K={K_layers}"):
        if u >= len(user_emb):
            continue

        gt_items = test_data[u]
        if len(gt_items) == 0:
            continue

        scores = np.dot(item_emb, user_emb[u])
        scores[train_data.get(u, [])] = -np.inf

        top_k = np.argsort(scores)[-20:][::-1]

        recall_list.append(recall_at_k(top_k, gt_items))
        ndcg_list.append(ndcg_at_k(top_k, gt_items))

    return {
        "recall@20": float(np.mean(recall_list)),
        "ndcg@20": float(np.mean(ndcg_list))
    }

# =========================
# RUN ALL MODELS
# =========================
base_path = r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\results\tables\MAIN TABLE\yelp_train"

models = {
    "k1": (os.path.join(base_path, "yelp_k1", "best_model.pt"), 1),
    "k2": (os.path.join(base_path, "yelp_k2", "best_model.pt"), 2),
    "k3": (os.path.join(base_path, "yelp_k3", "best_model.pt"), 3),
    "k4": (os.path.join(base_path, "yelp_k4", "best_model.pt"), 4),
}

results = {}

for key in models:
    path, k = models[key]
    results[key] = evaluate_model(path, k)

# =========================
# SAVE RESULTS
# =========================
with open("gowalla_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n===== FINAL RESULTS =====")
for k in results:
    print(k, results[k])