import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. LOAD LIGHTGCN-SINGLE
# =========================
single_files = {
    1: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\LIGHT-GCN-SINGLE\light-gcn_single_k1\lightgcn_single_outputs\results_k1.json",
    2: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\LIGHT-GCN-SINGLE\light-gcn_single_k2\lightgcn_single_outputs\results_k2.json",
    3: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\LIGHT-GCN-SINGLE\light-gcn_single_k3\lightgcn_single_outputs\results_k3.json",
    4: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\LIGHT-GCN-SINGLE\light-gcn_single_k4\lightgcn_single_outputs\results_k4.json",
}

single_results = {}
for k, path in single_files.items():
    with open(path, "r") as f:
        single_results[k] = json.load(f)

# =========================
# 2. LOAD LIGHTGCN (FULL)
# =========================
lgcn_files = {
    1: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\MAIN TABLE\Gowalla_train\k1_results\results_k1_gowalla.json",
    2: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\MAIN TABLE\Gowalla_train\k2_results\results_k2_gowalla.json",
    3: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\MAIN TABLE\Gowalla_train\k3_results\results_k3_gowalla.json",
    4: r"F:\SEM-6\ELECTIVES\DEPARTMENT-ELECTIVES\Recommendation Systems\Project\MAIN TABLE\Gowalla_train\k4_results\results_k4_gowalla.json",
}

lgcn_results = {}
for k, path in lgcn_files.items():
    with open(path, "r") as f:
        lgcn_results[k] = json.load(f)

# =========================
# 3. EXTRACT METRICS
# =========================
layers = sorted(single_results.keys())

recall_single = [single_results[k]["recall@20"] for k in layers]
ndcg_single   = [single_results[k]["ndcg@20"] for k in layers]

recall_lgcn = [lgcn_results[k]["recall@20"] for k in layers]
ndcg_lgcn   = [lgcn_results[k]["ndcg@20"] for k in layers]

# =========================
# 4. BAR POSITIONS
# =========================
x = np.arange(len(layers))
width = 0.35

# =========================
# 5. PLOT (MATCHES PAPER)
# =========================
plt.figure(figsize=(10,4))

# ---- Recall Plot ----
plt.subplot(1,2,1)
plt.bar(x - width/2, recall_lgcn, width, label='LightGCN')
plt.bar(x + width/2, recall_single, width, label='LightGCN-single')

plt.xticks(x, layers)
plt.xlabel("Number of Layers")
plt.ylabel("Recall@20")
plt.title("Gowalla")
plt.legend()

# ---- NDCG Plot ----
plt.subplot(1,2,2)
plt.bar(x - width/2, ndcg_lgcn, width, label='LightGCN')
plt.bar(x + width/2, ndcg_single, width, label='LightGCN-single')

plt.xticks(x, layers)
plt.xlabel("Number of Layers")
plt.ylabel("NDCG@20")
plt.title("Gowalla")
plt.legend()

plt.tight_layout()
plt.show()