import copy
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle  # using pickle instead of BSON for serialization

# These are assumed to be defined/imported from your codebase.
# from my_module import (random_rca_dag, sample_noise, forward, forward_scaled, zval,
#                        BayesNet, RootCPD, MlpCPD, LinearCPD, LinearBayesianCPD, MixedDist,
#                        Normal, Laplace)

# For demonstration, we define stubs:
def random_rca_dag(min_depth, n_nodes, n_root_nodes):
    # Stub: return an object with attributes:
    #   - adjacency_matrix: a NumPy array (shape: [n_nodes, n_nodes])
    #   - nv(): returns number of nodes.
    class DummyDAG:
        def __init__(self, n):
            self.adjacency_matrix = np.random.randint(0, 2, size=(n, n)).astype(bool)
            self.n = n
        def nv(self):
            return self.n
    return DummyDAG(n_nodes)

def sample_noise(g, n_samples):
    # Stub: returns a NumPy array of noise samples.
    return np.random.randn(g.nv(), n_samples)

def forward(g, epsilon):
    # Stub: compute forward pass (data = mean + noise)
    # Here we simply return noise shifted by 1 for demonstration.
    return epsilon + 1.0

def forward_scaled(cpd, x, mu, sigma):
    # Stub: apply the CPD’s scaling (for plotting, etc.)
    return x  # no change for now

def zval(d, sample):
    # Stub: compute a “z‐value” for a distribution d and sample.
    # Assume d has attributes mean and scale.
    # Here we use: z = (sample - mean)/scale.
    return (sample - d.mean) / d.scale

# Dummy CPD and BayesNet classes:
class BayesNet:
    def __init__(self):
        self.cpds = []
        self.name_to_index = {}
        # We assume the DAG is stored as an attribute:
        self.dag = type("DummyDAGContainer", (), {"adjacency_matrix": None})
    def add_cpd(self, cpd):
        self.cpds.append(cpd)
        self.name_to_index[cpd.target] = len(self.cpds) - 1
    def __getitem__(self, node):
        return self.cpds[self.name_to_index[node]]

class RootCPD:
    def __init__(self, target, noise_dists):
        self.target = target
        self.noise_dists = noise_dists
        self.parents = []
        self.d = noise_dists  # for simplicity

class MlpCPD:
    def __init__(self, target, parents, mlp, noise_dists):
        self.target = target
        self.parents = parents
        self.mlp = mlp
        self.noise_dists = noise_dists
        self.d = noise_dists  # for simplicity

class LinearCPD:
    def __init__(self, target, parents, a, d):
        self.target = target
        self.parents = parents
        self.a = a
        self.d = d

class LinearBayesianCPD:
    def __init__(self, target, parents):
        self.target = target
        self.parents = parents

class MixedDist:
    def __init__(self, ds):
        self.ds = ds

# For Normal and Laplace, we assume a minimal interface:
class Normal:
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale
    def params(self):
        return self.mean, self.scale
    def __repr__(self):
        return f"Normal({self.mean}, {self.scale})"

class Laplace:
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale
    def params(self):
        return self.mean, self.scale
    def __repr__(self):
        return f"Laplace({self.mean}, {self.scale})"

# ----------------------------
# Function definitions in Python
# ----------------------------

def random_mlp_dag_generator(min_depth, n_nodes, n_root_nodes, hidden, noise_dists, activation):
    logging.info("random_nonlinear_dag_generator")
    dag_obj = random_rca_dag(min_depth, n_nodes, n_root_nodes)
    # Ensure the adjacency matrix is boolean
    B = dag_obj.adjacency_matrix.astype(bool)
    d = dag_obj.nv()

    bn = BayesNet()
    # Store the adjacency matrix in the BayesNet for later use (if needed)
    bn.dag.adjacency_matrix = B

    node_names = [f"X{i}" for i in range(1, d+1)]

    for j in range(d):
        if np.sum(B[:, j]) == 0:  # root node
            cpd = RootCPD(target=node_names[j], noise_dists=noise_dists)
            bn.add_cpd(cpd)
        else:
            parent_indices = np.where(B[:, j])[0]
            parents = [node_names[i] for i in parent_indices]
            pa_size = len(parents)
            # Generate weights W1 and W2
            W1 = np.random.uniform(0.5, 2.0, size=(hidden, pa_size))
            mask1 = np.random.rand(*W1.shape) < 0.5
            W1[mask1] *= -1
            W2 = np.random.uniform(0.5, 2.0, size=(1, hidden))
            mask2 = np.random.rand(*W2.shape) < 0.5
            W2[mask2] *= -1
            # Create MLP using PyTorch (note: no bias as in Flux)
            layer1 = nn.Linear(pa_size, hidden, bias=False)
            layer2 = nn.Linear(hidden, 1, bias=False)
            # Build sequential network with activation in between
            mlp = nn.Sequential(layer1, activation, layer2).double()
            # Manually set weights
            with torch.no_grad():
                layer1.weight.copy_(torch.tensor(W1, dtype=torch.float64))
                layer2.weight.copy_(torch.tensor(W2, dtype=torch.float64))
            cpd = MlpCPD(target=node_names[j], parents=parents, mlp=mlp, noise_dists=noise_dists)
            bn.add_cpd(cpd)
    return bn

def copy_linear_dag(bn):
    g = copy.deepcopy(bn)
    for i, cpd in enumerate(g.cpds):
        if cpd.parents:
            a = np.random.randn(len(cpd.parents))
            g.cpds[i] = LinearCPD(cpd.target, cpd.parents, a, cpd.d)
    return g

def copy_bayesian_dag(bn):
    g = copy.deepcopy(bn)
    for i, cpd in enumerate(g.cpds):
        if cpd.parents:
            g.cpds[i] = LinearBayesianCPD(cpd.target, cpd.parents)
    return g

# In Python, we use __getitem__ in the BayesNet class (see above).

def scale3(d, s=3):
    # If d is a MixedDist, apply scale3 to each element in its ds list.
    if isinstance(d, MixedDist):
        return MixedDist([scale3(di, s) for di in d.ds])
    else:
        # Assume d has a params() method that returns (mu, sigma)
        mu, sigma = d.params()
        # Create a new distribution of the same type with scaled sigma.
        return type(d)(mu, s * sigma)

def draw_normal_perturbed_anomaly(g, n_anomaly_nodes, args):
    d = len(g.cpds)
    # normal data
    epsilon = sample_noise(g, args.n_samples)
    x = forward(g, epsilon)

    # perturbed data: scale root node distributions by 3σ
    g3 = copy.deepcopy(g)
    for cpd in g3.cpds:
        if not cpd.parents:  # perturb root nodes
            cpd.d = scale3(cpd.d)
    epsilon3 = sample_noise(g3, args.n_samples)
    x3 = forward(g3, epsilon3)

    # select anomaly nodes
    ga = copy.deepcopy(g)
    anomaly_nodes = np.random.choice(np.arange(d), size=n_anomaly_nodes, replace=False)
    for a in anomaly_nodes:
        ga.cpds[a].d = scale3(ga.cpds[a].d)

    # anomaly data: note the factor 20 multiplied by number of anomaly samples
    epsilon_a = sample_noise(ga, 20 * args.n_anomaly_samples)
    x_a = forward(ga, epsilon_a)

    # compute z-values for each distribution and each noise sample row
    # Assume epsilon_a is 2D with shape (features, n_samples)
    ds = [cpd.d for cpd in g.cpds]
    # Compute zval for each distribution (row) over each sample (column)
    za = np.array([[zval(distr, sample) for sample in epsilon_a.T] for distr in ds])

    # For anomaly nodes, get z-values and select indices where min(abs(z)) > 3
    z2 = za[anomaly_nodes, :]
    condition = np.min(np.abs(z2), axis=0) > 3
    ia = np.where(condition)[0]
    ia = ia[:args.n_anomaly_samples]

    epsilon_a = epsilon_a[:, ia]
    x_a = x_a[:, ia]

    y = x - epsilon
    y3 = x3 - epsilon3
    y_a = x_a - epsilon_a

    return epsilon, x, y, epsilon3, x3, y3, epsilon_a, x_a, y_a, anomaly_nodes

def plot1(j, g, epsilon, x, y, mu_x, sigma_x, epsilon_a, x_a, y_a, anomaly_nodes):
    logging.info(f"Plotting node {j}")
    cpd = g.cpds[j]
    # Get the j-th column of the DAG adjacency matrix as boolean mask
    paj = g.dag.adjacency_matrix[:, j]

    def parent_child(arr):
        # Returns tuple: (data for parent nodes, data for the child node)
        return arr[paj, :], arr[np.array([j]), :]

    # Apply parent_child to all inputs
    eps_parents, _ = parent_child(epsilon)
    x_parents, _ = parent_child(x)
    y_parents, _ = parent_child(y)
    mu_parents, _ = parent_child(mu_x)
    sigma_parents, _ = parent_child(sigma_x)

    epsa_parents, _ = parent_child(epsilon_a)
    xa_parents, _ = parent_child(x_a)
    ya_parents, _ = parent_child(y_a)

    fig = plot23d_pca(cpd, eps_parents, x_parents, y_parents, mu_parents, sigma_parents,
                      epsa_parents, xa_parents, ya_parents, j, anomaly_nodes)
    return fig

def plot23d_pca(cpd, eps, xs, ys, mu, sigma, eps_a, xs_a, ys_a, j, anomaly_nodes):
    # Compute forward_scaled predictions (assumed defined)
    y_pred = forward_scaled(cpd, xs, mu, sigma)
    ya_pred = forward_scaled(cpd, xs_a, mu, sigma)

    # Set default limits (if desired)
    xlim = ylim = (-5, 5)
    title_suffix = "abnormal" if j in anomaly_nodes else ""
    title = f"f_{{{j}}} {title_suffix}"

    # If input has more than 2 dimensions, apply PCA to reduce to 2D.
    if xs.shape[0] > 2:
        logging.info("More than 2D: applying PCA")
        title = f"PCA {j} {title_suffix}"
        pca = PCA(n_components=2)
        xs = pca.fit_transform(xs.T).T  # transform and take transpose to retain shape (2, n_samples)
        xs_a = pca.transform(xs_a.T).T

    # Use first row for x and second for output if available
    x12 = xs[0, :]
    # For plotting a base line (here zeros)
    z3 = np.zeros_like(x12)
    xa12 = xs_a[0, :]
    # For output, assume second row (if exists) else use prediction
    x3 = xs[1, :] if xs.shape[0] > 1 else y_pred
    xa3 = xs_a[1, :] if xs_a.shape[0] > 1 else ya_pred

    # Set up a scatter plot
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    fig, ax = plt.subplots()
    ax.scatter(x12, z3, s=2, label="input", color=colors[0])
    ax.scatter(xa12, z3, s=2, label="input outliers", color=colors[1])
    ax.scatter(x12, x3, s=2, label="output", color=colors[2])
    ax.scatter(xa12, xa3, s=2, label="output outliers", color=colors[3])
    ax.scatter(x12, y_pred, s=2, label="output mean", color=colors[4])
    ax.scatter(xa12, ya_pred, s=2, label="output mean outliers", color=colors[5])
    ax.set_title(title)
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig

def dist_name(d):
    # Returns a string combining the distribution type and its scale.
    s = type(d).__name__
    return f"{s}{d.scale}"

def data_path_from_ds(ds):
    # ds is assumed to be a list of distributions (e.g. MixedDist.ds)
    names = "".join([dist_name(d) for d in ds])
    return f"data/data-{names}.pkl"

def fig_name(args):
    return f"{args.noise_dist}-{args.data_id}"

def generate_data_skewed(args):
    # Unpack arguments from args (assumed to have attributes: min_depth, n_nodes, n_root_nodes,
    # n_anomaly_nodes, noise_dist, hidden, activation, etc.)
    for _ in range(5):
        # Generate two random distributions (choosing between Normal and Laplace)
        # For scales, choose from np.linspace(0.1, 1.0, 10)
        scales = np.random.choice(np.linspace(0.1, 1.0, 10), size=2, replace=True)
        ds = [np.random.choice([Normal, Laplace])(0, s) for s in scales]
        noise_dists = MixedDist(ds)
        file_path = data_path_from_ds(noise_dists.ds)
        logging.info("Generating " + file_path)
        g = random_mlp_dag_generator(args.min_depth, args.n_nodes, args.n_root_nodes,
                                     args.hidden, noise_dists, args.activation)
        results = draw_normal_perturbed_anomaly(g, args.n_anomaly_nodes, args)
        # Unpack results
        epsilon, x, y, epsilon3, x3, y3, epsilon_a, x_a, y_a, anomaly_nodes = results
        logging.info("Anomaly nodes: " + str(anomaly_nodes))
        # Save the data to file using pickle
        data_dict = {
            'args': args,
            'g': g,
            'epsilon': epsilon,
            'x': x,
            'y': y,
            'epsilon3': epsilon3,
            'x3': x3,
            'y3': y3,
            'epsilon_a': epsilon_a,
            'x_a': x_a,
            'y_a': y_a,
            'anomaly_nodes': anomaly_nodes,
            'ds': ds
        }
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

def generate_data_timeit(args):
    # Create a range for n_nodes: exponential spacing between 50 and 1000
    rng = np.exp(np.linspace(np.log(50), np.log(1000), 30))
    rng = np.round(rng).astype(int)
    for n_nodes in rng:
        for id in range(1, 6):
            scales = np.random.choice(np.linspace(0.1, 1.0, 10), size=2, replace=True)
            ds = [np.random.choice([Normal, Laplace])(0, s) for s in scales]
            noise_dists = MixedDist(ds)
            logging.info("Generating " + data_path_from_ds(noise_dists.ds))
            g = random_mlp_dag_generator(args.min_depth, n_nodes, args.n_root_nodes,
                                         args.hidden, noise_dists, args.activation)
            results = draw_normal_perturbed_anomaly(g, args.n_anomaly_nodes, args)
            epsilon, x, y, epsilon3, x3, y3, epsilon_a, x_a, y_a, anomaly_nodes = results
            logging.info("Anomaly nodes: " + str(anomaly_nodes))
            filepath = f"data/data-timeit&nodes={n_nodes}&id={id}.pkl"
            data_dict = {
                'args': args,
                'g': g,
                'epsilon': epsilon,
                'x': x,
                'y': y,
                'epsilon3': epsilon3,
                'x3': x3,
                'y3': y3,
                'epsilon_a': epsilon_a,
                'x_a': x_a,
                'y_a': y_a,
                'anomaly_nodes': anomaly_nodes,
                'ds': ds
            }
            with open(filepath, "wb") as f:
                pickle.dump(data_dict, f)

def plot_data(args):
    # Load data from file; here we assume that args.noise_dist helps determine the file path.
    file_path = f"data/data-{args.noise_dist}.pkl"
    logging.info("Loading " + file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    g = data_dict['g']
    epsilon = data_dict['epsilon']
    x = data_dict['x']
    y = data_dict['y']
    epsilon3 = data_dict['epsilon3']
    x3 = data_dict['x3']
    y3 = data_dict['y3']
    epsilon_a = data_dict['epsilon_a']
    x_a = data_dict['x_a']
    y_a = data_dict['y_a']
    anomaly_nodes = data_dict['anomaly_nodes']

    # Normalize data along each row (axis=1)
    X = x
    mu_x = np.mean(X, axis=1, keepdims=True)
    sigma_x = np.std(X, axis=1, keepdims=True)
    normalise_x = lambda arr: (arr - mu_x) / sigma_x
    scale_epsilon = lambda eps: eps / sigma_x

    # Normalize in place
    x = normalise_x(x)
    x3 = normalise_x(x3)
    x_a = normalise_x(x_a)
    y = normalise_x(y)
    y3 = normalise_x(y3)
    y_a = normalise_x(y_a)
    epsilon = scale_epsilon(epsilon)
    epsilon3 = scale_epsilon(epsilon3)
    epsilon_a = scale_epsilon(epsilon_a)

    # Sanity checks
    assert np.allclose(x, y + epsilon)
    assert np.allclose(x3, y3 + epsilon3)
    assert np.allclose(x_a, y_a + epsilon_a)

    # Assume g.dag.adjacency_matrix is defined
    d = g.dag.adjacency_matrix.shape[0]
    x1 = x[0, :]
    xa1 = x_a[0, :]
    s = "abnormal" if 0 in anomaly_nodes else ""
    title = f"root {s}"

    fig1, ax1 = plt.subplots()
    ax1.scatter(x1, np.zeros_like(x1), s=2, label="input", color='b')
    ax1.scatter(xa1, np.zeros_like(xa1), s=2, label="input outliers", color='g')
    ax1.set_title(title)

    figs = []
    # In Julia the plotting is done for nodes 2:d; here we loop over indices 1 to d-1.
    for j in range(1, d):
        fig_j = plot1(j, g, epsilon, x, y, mu_x, sigma_x, epsilon_a, x_a, y_a, anomaly_nodes)
        figs.append(fig_j)

    # For a complete figure, you might want to arrange these subplots.
    # Here we simply save the first figure as an example.
    overall_fig_path = f"fig/fcm-outliers-{fig_name(args)}.png"
    fig1.savefig(overall_fig_path)
    logging.info("Saved figure to " + overall_fig_path)
    plt.show()

def load_normalised_data(args):
    fpaths = glob.glob("data/*.pkl")
    assert len(fpaths) > 0, "No data files found."
    fpath = fpaths[args.data_id]
    logging.info("Loading " + fpath)
    with open(fpath, "rb") as f:
        data_dict = pickle.load(f)

    g = data_dict['g']
    epsilon = data_dict['epsilon']
    x = data_dict['x']
    y = data_dict['y']
    epsilon3 = data_dict['epsilon3']
    x3 = data_dict['x3']
    y3 = data_dict['y3']
    epsilon_a = data_dict['epsilon_a']
    x_a = data_dict['x_a']
    y_a = data_dict['y_a']
    anomaly_nodes = data_dict['anomaly_nodes']

    X = x
    mu_x = np.mean(X, axis=1, keepdims=True)
    sigma_x = np.std(X, axis=1, keepdims=True)
    normalise_x = lambda arr: (arr - mu_x) / sigma_x
    scale_epsilon = lambda eps: eps / sigma_x

    x = normalise_x(x)
    x3 = normalise_x(x3)
    x_a = normalise_x(x_a)
    y = normalise_x(y)
    y3 = normalise_x(y3)
    y_a = normalise_x(y_a)
    epsilon = scale_epsilon(epsilon)
    epsilon3 = scale_epsilon(epsilon3)
    epsilon_a = scale_epsilon(epsilon_a)

    assert np.allclose(x, y + epsilon)
    assert np.allclose(x3, y3 + epsilon3)
    assert np.allclose(x_a, y_a + epsilon_a)

    return g, x, x3, x_a, y, y3, y_a, epsilon, epsilon3, epsilon_a, mu_x, sigma_x, anomaly_nodes

