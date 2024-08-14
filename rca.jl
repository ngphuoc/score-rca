include("imports.jl")

using Revise
using DataFrames, Distributions, BayesNets, CSV, Tables
using BayesNets: plot, name
using DataFrames: index
using Graphs, GraphPlot
using Revise
using DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2
using Optimisers, BSON
using ProgressMeter: Progress, next!
using CUDA
using Flux
using Flux: gpu, Chain, Dense, relu, DataLoader

include("bayesnets-extra.jl")
include("group-mlp-unet.jl")
include("group-mlp-regression.jl")
include("lib/diffusion.jl")

function df_(pydf)
    @> pydf PyPandasDataFrame DataFrame
end

function create_score_model_from_ground_truth_dag(g0, all_nodes, training_data; args)
    d = length(all_nodes)
    B = adjacency_matrix(g0)
    # paj_mask = @> B Matrix{Bool} gpu
    paj_mask = @> B Matrix{Float32} gpu
    unet = GroupMlpUnet(B) |> gpu
    mlp = GroupMlpRegression(B) |> gpu
    X = @> df_(training_data) Array;
    opt = Flux.setup(Optimisers.AdamW(args.lr, (0.9, 0.999), args.decay), unet);
    # AdamW(η = 0.001, β = (0.9, 0.999), λ = 0, ϵ = 1e-8)
    loader = DataLoader((X',); args.batchsize, shuffle=true)
    (x,) = loader |> first;
    @≥ x gpu;
    t = rand!(similar(x));
    conditional_score_matching_loss(mlp, unet, x, paj_mask)

    loss, (grad,) = Flux.withgradient(unet, ) do unet
        conditional_score_matching_loss(mlp, unet, x, paj_mask)
    end

    progress = Progress(args.epochs, desc="Fitting unet")
    for epoch = 1:args.epochs
        total_loss = 0.0
        for (x,) = loader
            @≥ x gpu;
            t = rand!(similar(x));
            loss, (grad,) = Flux.withgradient(unet, ) do unet
                conditional_score_matching_loss(mlp, unet, x, paj_mask)
            end
            Flux.update!(opt, unet, grad)
            total_loss += loss
        end
        next!(progress; showvalues=[(:loss, total_loss/length(loader))])
    end

    return unet
end

"""
model_type=NonlinearScoreCPD
"""
function create_model_from_ground_truth_dag(g0, all_nodes, training_data; args, model_type=NonlinearGaussianCPD)
    d = length(all_nodes)
    B = adjacency_matrix(g0)
    g = DiGraph(B)
    cpds = map(1:d) do j
        ii = parent_indices(B, j)
        node = all_nodes[j]
        if length(ii) == 0
            # fit(StaticCPD{Normal}, df_(training_data), node)
            cpd = StaticCPD(node, Normal(0, 1))
        else
            pa = all_nodes[ii]
            # fit(LinearBayesianCPD, df_(training_data), node, pa)
            cpd = fit(model_type, df_(training_data), node, pa; args)
            cpd
        end
    end
    bn0 = BayesNet(cpds; use_topo_sort=false)

    return bn0
end

# learned_dag = fit_dag!(pydeepcopy(ground_truth_dag), pytable(normal_samples))
function fit_dag!(bn::BayesNet{CPD}, normal_samples)
    cpds = map(bn.cpds) do cpd
        @show typeof(cpd)
        node = name(cpd)
        pa = parents(cpd)
        if length(pa) == 0
            fit(StaticCPD{Normal}, normal_samples, node)
        else
            fit(NonlinearGaussianCPD, normal_samples, node, pa)
        end
    end
    bn0 = BayesNet(cpds; use_topo_sort=false)
    return bn0
end

"""
Use a list of fcms in the topo order to create a BayesSEM
Also filter observations in the node subset.
"""
function create_model_from_sub_graph(g0, sub_nodes, all_nodes, df_normal_samples; args)
    d = length(sub_nodes)
    ii = indexin(sub_nodes, all_nodes)
    B = adjacency_matrix(g0)[ii, ii]
    g = DiGraph(B)
    cpds = map(1:d) do j
        ii = parent_indices(B, j)
        node = sub_nodes[j]
        if length(ii) == 0
            fit(StaticCPD{Normal}, df_normal_samples, node)
        else
            pa = sub_nodes[ii]
            fit(NonlinearGaussianCPD, df_normal_samples, node, pa; args)
        end
    end
    bn = BayesNet(cpds; use_topo_sort=false)
    return bn
end

"""
Use a list of fcms in the topo order to create a BayesSEM
Also filter observations in the node subset.
Use the provided w_noise_scale to scale all weights w
"""
function create_model_from_sub_graph_with_w_noise_scale(g0, sub_nodes, all_nodes, w_noise_scale)
    d = length(sub_nodes)
    ii = indexin(sub_nodes, all_nodes)
    B = adjacency_matrix(g0)[ii, ii]
    g = DiGraph(B)
    cpds = map(1:d) do j
        ii = parent_indices(B, j)
        node = sub_nodes[j]
        if length(ii) == 0
            fit(StaticCPD{Normal}, df_normal_samples, node)
        else
            pa = sub_nodes[ii]
            # print("fit(LinearBayesianCPD, df_normal_samples, node, pa, w_noise_scale)")
            @show node, pa
            fit(LinearBayesianCPD, df_normal_samples, node, pa, w_noise_scale)
        end
    end
    bn = BayesNet(cpds; use_topo_sort=false)
    return bn
end

function fit_dag!(dag, training_data)
    for node in dag.graph.nodes
        if is_root_node(dag.graph, node) |> pytruth
            dag.set_causal_mechanism(node, EmpiricalDistribution())
        else
            dag.set_causal_mechanism(node, AdditiveNoiseModel(create_linear_regressor()))
        end
    end
    # Fit causal mechanisms..
    gcm.fit(dag, training_data)
    return dag
end

function zeros_ξs(bn)
    map(cpd->zeros(length(cpd.parents)), bn.cpds)
end

"Return the SEM version given the weight noise ξ"
function get_bayes_sem_W(bn; ξs = zeros_ξs(bn))
    A = adjacency_matrix(bn.dag)
    W = zeros(size(A))
    d = size(W, 1)
    j = 4
    edges = CartesianIndex{2}[]
    for j=1:d
        cpd, ξ = bn.cpds[j], ξs[j]
        # ii = indexin(cpd.parents, nodes)
        ii = getindex.([bn.name_to_index], cpd.parents)
        if length(ii) > 0
            W[ii, j] .= cpd.ps[1] .+ ξ
            append!(edges, CartesianIndex{2}[CartesianIndex(i, j) for i in ii])
        end
    end
    return W, edges
end

function select_outlier_node(bn0, sub_nodes; n_outlier_nodes)
    anomaly_nodes = []
    if n_outlier_nodes > 0
        anomaly_nodes = sample(sub_nodes, n_outlier_nodes, replace=false)
    end
    return anomaly_nodes
end

function assign_outlier_noise!(outlier_bn0, anomaly_nodes)
    if length(anomaly_nodes) > 0
        anomaly_cpds = get.([outlier_bn0], anomaly_nodes)
        for cpd in anomaly_cpds
            a = rand(rand() > .5 ? Uniform(5, 10) : -Uniform(5, 10))
            b = rand(Uniform(5, 10))
            if cpd isa StaticCPD
                @unpack μ, σ = cpd.d
                cpd.d = Normal(a, b)
            else
                cpd.b = rand(Normal(a, b))
            end
        end
    end
end

function create_outlier_bn(bn0, sub_nodes; n_outlier_nodes, n_outlier_edges)
    anomaly_nodes, anomaly_edges = select_outlier_node_edge(bn0, sub_nodes; n_outlier_nodes, n_outlier_edges)
    outlier_bn = deepcopy(bn0)
    assign_outlier_noise!(outlier_bn, anomaly_nodes, anomaly_edges; node_noise_type, edge_noise_type)
    return (outlier_bn, anomaly_nodes, anomaly_edges)
end

"Assuming diagonal covariance matrix, i.e. independent weights"
function get_w(cpd::LinearBayesianCPD, ξ)
    w = cpd.ps[1] + ξ
    return w
end

function get_w(cpd::LinearGaussianCPD, args...)
    cpd.a
end

get_σ(cpd::LinearBayesianCPD) = 1 / cpd.ps[end]

function get_w(cpd::StaticCPD, ξ)
    Float64[]
end

"Return the SEM version given the weight noise ξ"
function get_sem(bn, ξs)
    cpds = map(bn.cpds, ξs) do cpd, ξ
        isa(cpd, LinearBayesianCPD) ?
        LinearGaussianCPD(cpd.target, cpd.parents, cpd.ps[1] + ξ, 0, cpd.ps[4]) :
        cpd
    end
    BayesNet(cpds; use_topo_sort=false)
end

function deterministic(bn, ξs)
    get_sem(bn, ξs)
end

function CPDs.posterior(bn::BayesNet, df::DataFrame)
    cpds = map(bn.cpds) do cpd
        X = df[:, cpd.parents] |> Array{Float64}
        y = df[:, cpd.target] |> Array{Float64}
        cpd2 = deepcopy(cpd)
        if length(X) > 0
            stats = get_lr_stats(X, y)
            cpd2.ps = posterior(cpd.m, cpd.ps, stats)
        else
            stats = get_univar_stats(y)
            cpd2.d = posterior(cpd.d, stats)
        end
        return cpd2
    end
    return BayesNet(cpds; use_topo_sort=false)
end

function get_ξ_indices(bn)
    ξs = zeros_ξs(bn)
    ls = length.(ξs)
    e = cumsum(ls)
    b = circshift(e, 1)
    map((b,e)->collect(b+1:e), b, e)
end

function pack_ξ(ξs)
    vcats(ξs)
end

function unpack_ξ(ξ)
    getindex.([ξ], get_ξ_indices(bn))
end

""" Use maximum a posterior
Return:
- a list of noise ξs for use in bn
- the compact version ξ for use in bn
"""
function infer_ξ(bn, df_outlier_samples)
    bn_pos = posterior(bn, df_outlier_samples)
    W0, edges_ci = get_bayes_sem_W(bn)
    Wn, edges_ci = get_bayes_sem_W(bn_pos)
    ξ = Wn - W0
    return ξ[edges_ci]
end

"Inverse node noise"
function infer_ϵ(bn, X; ξs=zeros_ξs(bn))
    W, = get_bayes_sem_W(bn; ξs)
    ϵ = X - X*W
    return ϵ
end

function outlier_score(bn, target, x)
    cpd = get_cpd(bn, target)
    μ, σ = cpd.ps
    tail_prob = cdf(cpd, -abs(x - μ)/σ)
    -log(tail_prob)
end

function ndcg_scores(xs, ys; k=n_outliers)
    @show size(xs), size(ys)
    @> [ndcg_score(xs[[i], :], ys[[i], :]; k) for i=1:size(xs, 1)] float.()
end

function pack_ξ(ξs)
    vcats(ξs)
end

function unpack_ξ(ξ, bn)
    getindex.([ξ], get_ξ_indices(bn))
end

function nothing2empty(x)
    x === nothing ? Float64[] : x
end

function nothing2empties(xs::AbstractArray)
    nothing2empty.(xs)
end

function ndcg_scores(xs, ys; k=n_outliers)
    @show size(xs), size(ys)
    @> [ndcg_score(xs[[i], :], ys[[i], :]; k) for i=1:size(xs, 1)] float.()
end

