using ExponentialAction: parameters
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
# @py import pandas as pd

function get_ground_truth_dag(min_depth, n_nodes; scale=0.1, hidden=2)
    is_sufficiently_deep_graph = false
    ground_truth_dag = target_node = nothing
    while !is_sufficiently_deep_graph
        n_downstream_nodes = n_nodes - n_root_nodes
        # Generate DAG with random number of nodes and root nodes
        # ground_truth_dag = random_linear_dag_generator(n_root_nodes, n_downstream_nodes)
        ground_truth_dag = random_nonlinear_dag_generator(n_root_nodes, n_downstream_nodes, scale, hidden)
        # Make sure that the randomly generated DAG is deep enough.
        for node = sample(collect(ground_truth_dag.graph.nodes), length(collect(ground_truth_dag.graph.nodes)), replace=false)
            target_node = node
            if length(collect(networkx.ancestors(ground_truth_dag.graph, target_node))) + 1 >= min_depth
                is_sufficiently_deep_graph = true
                break
            end
        end
    end
    return ground_truth_dag, target_node
end

function ref_topo_sorted_dag_weights(causal_graph)
    graph = causal_graph.graph
    d = length(graph.nodes)
    W = zeros(d, d)
    ordered_nodes = collect(networkx.topological_sort(graph))
    ws = []  # edges
    ps = []  # noise dist.
    causal_models = []
    pas = []  # sets of PA_j
    for (j, node) = enumerate(ordered_nodes)
        if !pytruth(is_root_node(causal_graph.graph, node))
            # connect to parent node
            causal_model = causal_graph.causal_mechanism(node)
            w_j = causal_model.prediction_model.sklearn_model.coef_ |> PyArray
            parent_nodes = get_ordered_predecessors(causal_graph.graph, node) |> collect
            ii = indexin(parent_nodes, ordered_nodes)
            W[ii, j] .= w_j
            push!(ws, w_j)
            push!(ps, causal_model.noise_model)
            push!(causal_models, causal_model)
            push!(pas, parent_nodes)
        else
            causal_model = causal_graph.causal_mechanism(node)
            push!(ws, [])
            push!(ps, causal_model)
            push!(causal_models, causal_model)
            push!(pas, [])
        end
    end
    W, ordered_nodes, ws, ps, causal_models, pas
end

function create_outlier_dag(ground_truth_dag, target_node)
    # Create references to the outlier_dag
    outlier_dag = pydeepcopy(ground_truth_dag)
    W, ordered_nodes, ws, ps, causal_models, pas = ref_topo_sorted_dag_weights(outlier_dag)

    # outlier ancestors, and data
    pred_method, topo_path = gcm._noise.get_noise_dependent_function(outlier_dag, target_node)
    path_indices = indexin(PyList(topo_path), ordered_nodes)

    function in_path(node::Py)
        string(node) ∈ string.(topo_path)
    end

    path_indices = indexin(PyList(topo_path), ordered_nodes)
    function in_path(j::Int)
        j ∈ path_indices
    end

    function i_(node)  # index into ordered_nodes
        findfirst(==(string(node)), string.(ordered_nodes))
    end
    @assert i_(ordered_nodes[end]) == length(ordered_nodes)

    # p = ps[end]
    # m = causal_models[end]
    node_truths = edge_truths = Int[]

    if n_outlier_nodes > 0
        # sample nodes for outliers
        anomaly_nodes = sample(collect(topo_path), n_outlier_nodes, replace=false)
        node_truths = binary_vec(anomaly_nodes, PyList(topo_path))  # ground truths
        # outlier distribution
        for node = anomaly_nodes
            ps[i_(node)].parameters["loc"] = 3 * ps[i_(node)].parameters["scale"]
        end
    end

    edges = []
    for j = 1:length(ws)
        if in_path(j)
            for i = 1:length(ws[j])
                push!(edges, (j, i))
            end
        end
    end

    if n_outlier_edges > 0
        anomaly_edges = sample(edges, n_outlier_nodes, replace=false)
        edge_truths = binary_vec(anomaly_edges, edges)
        for (j, i) = anomaly_edges
            ws[j][i] = 3 * maximum(W)
        end
    end

    return outlier_dag, ordered_nodes, ws, ps, causal_models, pas, topo_path, node_truths, edge_truths
end

function get_node_scores(contributions)
    node_scores = @> map(contributions) do c
        dict = PyDict(c)
        node_scores = float.(getindex.([dict], PyList(topo_path)))
    end hcats transpose Array
end

function t_(node)  # index into topo_path
    findfirst(==(string(node)), string.(topo_path))
end
# @assert t_(PyList(topo_path)[end]) == length(topo_path)

function get_edge_scores(node_scores)
    scores = Float64[]
    for (j, i) = edges
        # @show j, i
        Xj = ordered_nodes[j]
        Xi = pas[j][i]
        push!(scores, node_scores[t_(Xj)] * node_scores[t_(Xi)])
    end
    return scores
end
# contributions = causal_rca_contributions
# c = contributions[1]

function get_data(min_depth, n_nodes; scale, hidden)
    ground_truth_dag, target_node = get_ground_truth_dag(min_depth, n_nodes; scale, hidden)
    training_data = draw_samples(ground_truth_dag, n_samples)
    ordered_nodes = collect(networkx.topological_sort(ground_truth_dag.graph))
    pred_method, topo_path = gcm._noise.get_noise_dependent_function(ground_truth_dag, target_node)
    return ground_truth_dag, target_node, training_data, ordered_nodes, topo_path
end

function df2pydf(df)
    @> Dict(k => df[!, k] for k in names(df)) PyDict pd.DataFrame
end

# pd.DataFrame(PyDict(Dict("a" => [1], "b" => [2])))
function df_(pydf)
    @> pydf PyPandasDataFrame DataFrame
end

