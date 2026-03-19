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
# LOAD DATA (SAME AS TRAIN)
# =========================
def load_data(path):
    data = defaultdict(set)
    with open(path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                continue
            u = parts[0]
            for i in parts[1:]:
                data[u].add(i)
    return data

train_data = load_data(r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\data\yelp2018\train.txt")
test_data  = load_data(r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\data\yelp2018\test.txt")

num_users = max(max(train_data.keys()), max(test_data.keys())) + 1
num_items = max(max([max(v) for v in train_data.values()]),
                max([max(v) for v in test_data.values()])) + 1

print(f"Users: {num_users}, Items: {num_items}")

# =========================
# EXACT SAME NORMALIZATION
# =========================
def build_norm_adj(train_dict, norm_mode):
    user_deg = np.zeros(num_users)
    item_deg = np.zeros(num_items)

    edges_u, edges_i = [], []

    for u, items in train_dict.items():
        user_deg[u] = len(items)
        for i in items:
            item_deg[i] += 1
            edges_u.append(u)
            edges_i.append(i)

    rows, cols, vals = [], [], []

    for u, i in zip(edges_u, edges_i):
        du, di = user_deg[u], item_deg[i]
        if du == 0 or di == 0:
            continue

        ui = num_users + i

        if norm_mode == "sym":
            w_ui = 1 / (np.sqrt(du) * np.sqrt(di))
            w_iu = w_ui

        elif norm_mode == "l":
            w_ui = 1 / np.sqrt(du)
            w_iu = 1 / np.sqrt(di)

        elif norm_mode == "r":
            w_ui = 1 / np.sqrt(di)
            w_iu = 1 / np.sqrt(du)

        elif norm_mode == "l1":
            w_ui = 1 / (du * di)
            w_iu = w_ui

        elif norm_mode == "l1_l":
            w_ui = 1 / du
            w_iu = 1 / di

        elif norm_mode == "l1_r":
            w_ui = 1 / di
            w_iu = 1 / du

        rows += [u, ui]
        cols += [ui, u]
        vals += [w_ui, w_iu]

    A = sp.coo_matrix((vals, (rows, cols)),
                      shape=(num_users + num_items, num_users + num_items))

    indices = torch.LongTensor([A.row, A.col])
    values = torch.FloatTensor(A.data)

    return torch.sparse_coo_tensor(indices, values, A.shape).coalesce().to(device)

# =========================
# MODEL (SAME)
# =========================
class LightGCN(torch.nn.Module):
    def __init__(self, n_users, n_items, dim, layers, A_norm):
        super().__init__()
        self.A_norm = A_norm
        self.n_layers = layers
        self.n_users = n_users

        self.user_emb = torch.nn.Embedding(n_users, dim)
        self.item_emb = torch.nn.Embedding(n_items, dim)

    def propagate(self):
        E = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_E = [E]

        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.A_norm, E)
            all_E.append(E)

        E = torch.stack(all_E, dim=1).mean(dim=1)
        return E[:self.n_users], E[self.n_users:]

    def get_embeddings(self):
        return self.propagate()

# =========================
# EVALUATION
# =========================
@torch.no_grad()
def evaluate(model):
    model.eval()
    u_emb, i_emb = model.get_embeddings()
    i_emb_T = i_emb.T

    recall, ndcg = 0, 0
    users = [u for u in test_data if len(test_data[u]) > 0]

    for u in tqdm(users):
        scores = torch.matmul(u_emb[u], i_emb_T).cpu().numpy()

        for ti in train_data.get(u, []):
            scores[ti] = -np.inf

        top_k = np.argpartition(-scores, 20)[:20]
        top_k = top_k[np.argsort(-scores[top_k])]

        gt = test_data[u]
        hits = len(set(top_k) & gt)
        recall += hits / min(len(gt), 20)

        dcg = sum(1/np.log2(i+2) for i, x in enumerate(top_k) if x in gt)
        idcg = sum(1/np.log2(i+2) for i in range(min(len(gt),20)))
        ndcg += dcg/idcg if idcg>0 else 0

    n = len(users)
    return recall/n, ndcg/n

# =========================
# RUN ALL MODELS
# =========================
base_dirs = [
    r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\results\tables\TABLE-5\Yelp\yelp_table5_part1",
    r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\RS-Project\results\tables\TABLE-5\Yelp\yelp_table5_part2"
]

results = {}

for base in base_dirs:
    for folder in os.listdir(base):
        path = os.path.join(base, folder)
        model_path = os.path.join(path, "best_model.pt")

        if not os.path.exists(model_path):
            continue

        checkpoint = torch.load(model_path, map_location=device)

        norm_mode = checkpoint["norm_mode"]   # 🔥 KEY FIX
        layers = checkpoint["n_layers"]

        print(f"\nEvaluating {folder} | norm={norm_mode}")

        A_norm = build_norm_adj(train_data, norm_mode)

        model = LightGCN(num_users, num_items, 64, layers, A_norm).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        r, n = evaluate(model)

        results[folder] = {
            "norm": norm_mode,
            "recall@20": float(r),
            "ndcg@20": float(n)
        }

# SAVE
with open("table5_eval_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n===== FINAL RESULTS =====")
for k,v in results.items():
    print(k, v)