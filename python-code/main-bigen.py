import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ndcg_score, classification_report, roc_auc_score, r2_score

# Dummy implementations to mimic Julia functions:
def load_normalised_data(args):
    """
    Expected to return:
       g, x, x3, xa, y, y3, ya, eps, eps3, epsa, mu_x, sigma_x, anomaly_nodes
    Here we create dummy tensors and a dummy graph.
    """
    # Dummy graph object with minimal attributes
    class DummyGraph:
        def __init__(self):
            self.cpds = "dummy_cpds"
            # For demonstration, create a dummy dag object with an adjacency_matrix
            class DummyDag:
                def __init__(self, d):
                    self.adjacency_matrix = (np.random.rand(d, d) > 0.5).astype(bool)
            self.dag = DummyDag(100)
    g = DummyGraph()
    # Create dummy tensors (assume 100 samples with 10 features each)
    x = torch.randn(100, 10)
    x3 = x.clone()
    xa = torch.randn(100, 10)
    # Make y such that x ≈ y + eps, with a small eps
    eps = torch.randn(100, 10) * 0.1
    y = x - eps
    y3 = y.clone()
    ya = y.clone()
    eps3 = eps.clone()
    epsa = eps.clone()
    mu_x = x.mean(dim=0)
    sigma_x = x.std(dim=0)
    anomaly_nodes = [1, 3, 5]  # dummy indices for anomaly nodes
    return g, x, x3, xa, y, y3, ya, eps, eps3, epsa, mu_x, sigma_x, anomaly_nodes

def copy_bayesian_dag(g):
    # Return a copy or similar Bayesian network object.
    # For this dummy example, we simply return g.
    return g

def forward_1step(model, x):
    # Dummy forward step: simply scale the input
    return x * 0.9

def forward_leaf(g, epsa, ii):
    # Dummy leaf function: sum the epsa tensor
    return epsa.sum()

# Dummy argument class to hold parameters:
class Args:
    def __init__(self):
        self.to_device = torch.device('cpu')  # or 'cuda' if available
        self.n_nodes = 100
        self.n_anomaly_nodes = 3
        self.hidden_dim = 64
        self.freeze = "none"
        self.resolution = "both"  # "8", "16", or "both"
        self.pretrained = False
        self.drop_rate = 0.1
        self.share_attention = False
        self.share_instance_prediction = False
        self.feature = "concat"
        self.noise_dist = "gaussian"
        self.data_id = 1
        self.min_depth = 3

args = Args()
to_device = args.to_device

# Load data and move to device
g, x, x3, xa, y, y3, ya, eps, eps3, epsa, mu_x, sigma_x, anomaly_nodes = load_normalised_data(args)
print("#-- 1. fit linear bayesnet")
assert torch.allclose(x, y + eps, atol=1e-6), "x is not approximately equal to y + eps"
z = x - y
# (Optional) reshape z if needed: z = z.view(-1, 1)

# Send tensors to the desired device:
z, x, x3, xa, y, y3, ya, eps, eps3, epsa, mu_x, sigma_x = [
    t.to(to_device) for t in [z, x, x3, xa, y, y3, ya, eps, eps3, epsa, mu_x, sigma_x]
]

bn = copy_bayesian_dag(g)
print("g.cpds:", getattr(g, 'cpds', None))
print("bn.cpds:", getattr(bn, 'cpds', None))
# Assume bn has a fit method; otherwise, replace with your fitting procedure:
if bn is not None and hasattr(bn, 'fit'):
    bn.fit(x)

print("#-- 2. define residual function as outlier scores")
mu_x = forward_1step(bn, x)

def get_residual(bn, x):
    mu_x = forward_1step(bn, x)
    return x - mu_x

eps_hat_x = get_residual(bn, x)
vx = torch.abs(eps_hat_x)  # anomaly_measure

print("#-- 3. ground truth ranking and results")
max_k = args.n_anomaly_nodes
overall_max_k = max_k + 1
# Assume g.dag has an adjacency_matrix attribute; otherwise, use a dummy matrix.
if hasattr(g, 'dag') and hasattr(g.dag, 'adjacency_matrix'):
    adjmat = g.dag.adjacency_matrix.astype(bool)
else:
    d = 100
    adjmat = (np.random.rand(d, d) > 0.5).astype(bool)
d = adjmat.shape[0]
ii = [np.where(col)[0] for col in adjmat.T]

def get_epsilon_rankings(epsa, grad_epsa):
    """
    epsa and grad_epsa: numpy arrays of shape (d, batchsize)
    Returns a list of score vectors (one per sample in the batch).
    """
    assert epsa.shape[0] == d
    scores = []
    batchsize = epsa.shape[1]
    for i in range(batchsize):
        tmp = {j: grad_epsa[j, i] * epsa[j, i] for j in range(d)}
        ranking = sorted(tmp, key=lambda j: tmp[j], reverse=True)
        score = np.zeros(d)
        for q in range(max_k):
            iq = ranking[q]
            score[iq] = overall_max_k - (q + 1)
        scores.append(score)
    return scores

# Compute gradient with respect to epsa using torch autograd:
epsa.requires_grad = True
leaf_out = forward_leaf(g, epsa, ii)
grad_epsa = torch.autograd.grad(leaf_out.sum(), epsa)[0]
gt_value = np.column_stack(get_epsilon_rankings(epsa.detach().numpy(), grad_epsa.detach().numpy()))
mean_gt_value = gt_value.mean(axis=1)
print("Anomaly nodes:", anomaly_nodes)

mu_a = forward_1step(g, xa)
eps_hat_a = xa - mu_a
anomaly_measure = torch.abs(get_residual(bn, xa))

# Create a ground truth mask for anomalies:
gt_manual = np.array([i in anomaly_nodes for i in range(d)], dtype=int)
gt_manual = np.repeat(gt_manual[:, np.newaxis], xa.shape[1], axis=1)

print("#-- 4. save results")
# df = pd.DataFrame(columns=[ "n_nodes", "n_anomaly_nodes", "method", "noise_dist", "data_id", "ndcg_ranking", "ndcg_manual", "k" ])
columns = ["n_nodes", "n_anomaly_nodes", "method", "noise_dist", "data_id", "ndcg_ranking", "ndcg_manual", "k"]
rows = []

for k in range(1, args.min_depth + 1):
    # ndcg_score expects arrays with shape (n_samples, n_labels).
    # Here we transpose gt_value and anomaly_measure so that each sample is a row.
    ndcg_ranking = ndcg_score(gt_value.T, anomaly_measure.detach().numpy().T, k=k)
    ndcg_manual = ndcg_score(gt_manual.T, anomaly_measure.detach().numpy().T, k=k)
    rows.append({
        "n_nodes": args.n_nodes,
        "n_anomaly_nodes": args.n_anomaly_nodes,
        "method": "Bayesian",
        "noise_dist": str(args.noise_dist),
        "data_id": args.data_id,
        "ndcg_ranking": ndcg_ranking,
        "ndcg_manual": ndcg_manual,
        "k": k
    })

# Create the DataFrame from the list of rows:
df = pd.DataFrame(rows, columns=columns)

print(df)
fname = "results/random-graphs.csv"
if not os.path.isfile(fname):
    df.to_csv(fname, index=False)
else:
    df.to_csv(fname, index=False, mode='a', header=False)

