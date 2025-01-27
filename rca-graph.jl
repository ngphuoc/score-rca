using PythonCall
using Parameters: @unpack
using Distributions: sample, mean, std
using DataFrames
using Flux: onehot
using Base: Fix2
# include("imports.jl")
include("lib/utils.jl")
include("lib/graph.jl")
include("lib/outlier.jl")
include("rca.jl")
include("bayesnets-extra.jl")
pyimport("sys").path.insert(0, "")
pyimport("sys").path.insert(0, "supp/scripts")
pyimport("sys").path

@py import pickle
@py import random
@py import networkx
@py import numpy
@py import pandas
@py import pybaseball as pyb
@py import scipy
@py import dowhy
@py import sklearn
@unpack stats = scipy
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")
tqdm = pyimport("tqdm").tqdm
pydeepcopy = pyimport("copy").deepcopy

@unpack plot_score_field, sample_2d = pyimport("data2d")

@unpack random_nonlinear_dag_generator, random_linear_dag_generator, draw_anomaly_samples, draw_samples_2, our_approach_rankings, naive_approach, MySquaredRegressor, get_noise_coefficient, evaluate_results_ndcg, summarize_result, get_ground_truth_rankings, ZOutlierScorePy, ShapleyApproximationMethods, pickle_save, pickle_load = pyimport("rca_helper")
@unpack draw_samples, is_root_node, EmpiricalDistribution, InvertibleStructuralCausalModel, ScipyDistribution, AdditiveNoiseModel = dowhy.gcm
@unpack gcm = dowhy
@unpack disable_progress_bars = dowhy.gcm.config
@unpack create_linear_regressor = dowhy.gcm.ml
@unpack get_ordered_predecessors = dowhy.gcm.graph
@unpack get_noise_dependent_function = dowhy.gcm._noise


function random_rca_dag(min_depth, n_nodes, n_root_nodes)
    is_sufficiently_deep_graph = false
    ground_truth_dag = target_node = nothing
    while !is_sufficiently_deep_graph
        n_downstream_nodes = n_nodes - n_root_nodes
        # Generate DAG with random number of nodes and root nodes
        ground_truth_dag = random_linear_dag_generator(n_root_nodes, n_downstream_nodes)
        # Make sure that the randomly generated DAG is deep enough.
        for node = sample(collect(ground_truth_dag.graph.nodes), length(collect(ground_truth_dag.graph.nodes)), replace=false)
            target_node = node
            if length(collect(networkx.ancestors(ground_truth_dag.graph, target_node))) + 1 >= min_depth
                is_sufficiently_deep_graph = true
                break
            end
        end
    end
    pred_method, topo_path = gcm._noise.get_noise_dependent_function(ground_truth_dag, target_node)
    all_nodes = networkx.nodes(ground_truth_dag.graph)
    target_node
    all_nodes, topo_path, target_node

    g, v = from_pydigraph(ground_truth_dag)
    topo_path = fmap(string, collect(topo_path))
    all_nodes = fmap(string, collect(all_nodes))
    target_node = string(target_node)
    ii = indexin(topo_path, all_nodes)
    target_node
    A = @> g adjacency_matrix Matrix{Bool}
    B = A[ii, ii]
    dag = DiGraph(B)
    topo_path
    target_node
    return dag
end

function test_rca_graph()
    min_depth = 4
    n_nodes = 40
    n_root_nodes = 1
    dag = random_rca_dag(min_depth, n_nodes, n_root_nodes)
    @> dag adjacency_matrix
end
