using Revise
using DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2
using Optimisers, BSON
using ProgressMeter: Progress, next!
include("./imports.jl")
include("rca.jl")
include("./data.jl")
include("./lib/diffusion.jl")

args = @env begin
    # Denoising
    input_dim = 2
    output_dim = 2
    hidden_dim = 32  # hiddensize factor
    embed_dim = 32  # hiddensize factor
    scale=30.0f0  # RandomFourierFeatures scale
    σ_max = 5.0
    σ_min = 1e-3
    lr = 1e-3  # learning rate
    decay = 1e-5  # weight decay parameter for AdamW
    to_device = gpu
    batchsize = 32
    epochs = 300
    save_path = "output"
    model_file = "2d-joint.bson"

    # RCA
    min_depth = 10  # minimum depth of ancestors for the target node
    n_root_nodes = 3  # num_root_nodes
    n_samples = 1000  # n observations
    n_outlier_samples = 8  # n faulty observations
    has_node_outliers = true  # node outlier setting
    has_edge_outliers = true  # edge outlier setting
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    noise_scale = 1.0
    seed = 1  #  random seed
end
include("random-graph-datasets.jl")

#--  CREATE DATASETS

n_nodes = round(Int, min_depth^2 / 3)
ground_truth_dag, target_node, training_data, ordered_nodes, topo_path = get_data(min_depth, n_nodes)
sub_nodes = Symbol.(collect(topo_path))
all_nodes = Symbol.(ordered_nodes)
target = Symbol(target_node)
d = length(topo_path)
g0, v0, i0_ = from_pydigraph(ground_truth_dag, ordered_nodes)
bn0 = create_model_from_ground_truth_dag(g0, all_nodes, training_data)
normal_samples = rand(bn0, min_depth*500)
learned_dag = fit_dag!(pydeepcopy(ground_truth_dag), pytable(normal_samples))
df_normal_samples = normal_samples[!, sub_nodes]
pred_method, topo_path = get_noise_dependent_function(learned_dag, target_node)
n_outliers = ceil(Int, 0.1d)
n = e = 0
has_node_outliers && (n = n_outliers)
has_edge_outliers && (e = n_outliers)
n_outlier_nodes, n_outlier_edges = n, e
anomaly_nodes, anomaly_edges = select_outlier_node_edge(bn0, sub_nodes; n_outlier_nodes, n_outlier_edges)

#--  RUN EXP

n_nodes0 = length(bn0.cpds)
n_samples0 = 100 * n_nodes0  # n observations
n_outlier_nodes, n_outlier_edges = length(anomaly_nodes), length(anomaly_edges)
n_outliers = n_outlier_nodes + n_outlier_edges
max_k = n_outlier_nodes + n_outlier_edges
overall_max_k = max_k + 1
bn = create_model_from_sub_graph(g0, sub_nodes, all_nodes, df_normal_samples)
n_nodes = length(bn.cpds)
n_samples = 100 * n_nodes  # n observations
W, edges_ci = get_bayes_sem_W(bn)
@> W minimum, maximum, mean, std

outlier_bn = deepcopy(bn0)
assign_outlier_noise!(outlier_bn, anomaly_nodes, anomaly_edges)
outlier_samples = rand(outlier_bn, n_outlier_samples)
df_outlier_samples = outlier_samples[!, sub_nodes]

