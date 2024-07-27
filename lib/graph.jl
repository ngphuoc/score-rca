include("utils.jl")
using Graphs, MetaGraphs, LinearAlgebra, Random, Distributions, Parameters, BayesNets
using Parameters: @unpack

function myplot(g, N)
    graphplot(g, names=N, fontsize=12, curves=false, nodeshape=:circle, markercolor=:white, markersize = 0.2)
end

trace = tr

is_dag(W::AbstractMatrix, ϵ=1e-8) = trace(exp(W .* W)) - size(W, 1) < ϵ

dag_reg(W::AbstractMatrix) = trace(exp(W .* W)) - size(W, 1)

# function dag_reg(A::AbstractMatrix)
#     d = size(A, 1)
#     A = A .* A
#     M = I + A ./ d  # (Yu et al. 2019)
#     E = power(M, d-1)
#     h = sum(E' * M) - d
#     return h
# end

const dag_constraint = dag_reg

# function is_dag(W::AbstractMatrix, ϵ=1e-8)
#     d = size(W, 1)
#     α = 1.0 / d
#     # (Yu et al. 2019: DAG-GNN; reformulation of original NOTEARS constraint)
#     M = I(d) + α .* W .* W
#     h = tr(M ^ d) - d
#     h = tr(M ^ d) - d
#     return h < ϵ
# end

"""
    rand_dag(rng, N, alpha = 0.1)
Create random DAG from randomly permuted random triangular matrix with
edge probability `alpha`.
"""
function rand_dag(rng::AbstractRNG, N, alpha = 0.1)
    g = DiGraph(N)
    p = randperm(rng, N)
    for i in 1:N
        for j in i+1:N
            rand(rng) < alpha && add_edge!(g, p[i], p[j])
        end
    end
    g
end

function random_permutation(rng::AbstractRNG, M)
    N = size(M, 1)
    P = I(N)[randperm(rng, N), :]
    return P * M * P'
end

"""Simulate random DAG with some expected number of edges.
Args
    D (int): num of nodes
    E (int): expected num of edges
    graph_type (str): ER, SF
Returns
    g (SimpleDiGraph): DAG graph
"""
function simulate_dag(rng::AbstractRNG, D, E, graph_type; permute=false)
    graph_type = lowercase(graph_type)
    p = E / (D * (D - 1) / 2)  # edge prob
    if graph_type ∈ ["er", "erdos"]
        g = erdos_renyi(D, p, is_directed=true, seed=rand(rng, UInt))
        a = @> adjacency_matrix(g) triu
    elseif graph_type ∈ ["sf", "scalefree"] # Scale-free, Barabasi-Albert
        k = round(Int, p*D/2)
        g = barabasi_albert(D, k, is_directed=true)
        a = @> adjacency_matrix(g) transpose
    else
        @error("unknown graph type")
    end
    permute && @≥₂ a random_permutation(rng)
    return a
end



# erdos_renyi(10, 0.5, is_directed=true, seed=B_rng) |> adjacency_matrix

"""Simulate SEM parameters for a DAG.
Args
    B (ndarray): [d, d] binary adj matrix of DAG
    w_ranges (tuple): disjoint weight ranges
Returns
    W (ndarray): [d, d] weighted adj matrix of DAG
"""
function simulate_parameters(rng::AbstractRNG, B, w_ranges=((0.5, 2.0), ))  # (-2.0, -0.5)
    W = zeros(size(B))
    d = size(W, 1)
    S = rand(rng, 1:length(w_ranges), d)  # which range
    for (i, (low, high)) in enumerate(w_ranges)
        U = rand(rng, Uniform(low, high), d)
        W += B .* (S .== i) .* U
    end
    return W
end

"""Equal var for now"""
function simulate_noisy_parameters(rng::AbstractRNG, B, w_ranges=((0.5, 2.0), ), σ2=0.2)  # (-2.0, -0.5)
    W = simulate_parameters(rng, B, w_ranges)
    return W, σ2
end

function init_prior(B)
    sz = size(B)
    (; μ = zeros(sz) , ρ = log.(1e-9ones(sz)) )
end

"""Assume topo sorted in the following functions"""

is_leaf(B, i) = sum(B[i, i+1:end]) == 0

function is_root(bn, node::Symbol)
    isa(get(bn, node), StaticCPD)
end

is_root(B, i) = is_leaf(B', i)

find_leaves(B) = map(Base.Fix1(is_leaf, B), 1:size(B, 1)) |> findall

find_root(B) = map(Base.Fix1(is_root, B), 1:size(B, 1)) |> findall

""" Intervene innernode weights (equal-var and 1-node for now)
    return W̃, (i, j)
"""
function random_intervention(rng::AbstractRNG, W, w_range=(5, 10))
    B = W .!= 0
    il = find_leaves(W)
    B[il, :] .= 0
    ci = findall(W .!= 0) |> rand
    i, j = ci.I
    ŵ = rand(rng, Uniform(w_range...))
    W̃ = copy(W)
    W̃[i, j] = ŵ
    return W̃, (i, j)
end

function reparam(rng, μ, σ)
    μ .+ σ .* randn(rng, size(σ))
end

# TODO
# - equal var for additive noise
# - may need to reduce link noise

function noisy_forward(rng, p, X)
    v_w = exp.(p.ρ)
    μ = p.μ'X
    v_w = exp.(p.ρ)
    σ² = v_w * X .^ 2
    Y = reparam(rng, μ, .√σ²)
    Y, (μ, σ²)
end

function noisy_generate(rng, p, n = 1000, noise_scale = 0.5)
    B = p.μ .!= 0
    d, = size(B)
    X = zeros(d, n)
    # ir = find_root(B)
    # X[ir, :] .= v_x .* rand(rng, length(ir), n)
    i = 1
    for i = 1:d
        Y, = noisy_forward(rng, p, X)
        ϵ = noise_scale * randn(rng, 1, n)
        X[[i], :] .= Y[[i], :] + ϵ # only use time step i
    end
    X
end

# ref: Graphs.jl
# Ref: https://www.geeksforgeeks.org/python-program-for-topological-sorting/
function topological_sort(A::AbstractMatrix{T}) where T
    d = size(A, 1)
    visited = zeros(Int, d)
    stack = Vector{Int}()
    for v in 1:d
        visited[v] != 0 && continue
        working_stack = [v]
        visited[v] = 1
        while !isempty(working_stack)
            u = working_stack[end]
            w = 0
            for N in outneighbors(A, u)
                if visited[N] == 1
                    error("The input graph contains at least one loop.") # TODO 0.7 should we use a different error?
                elseif visited[N] == 0
                    w = N
                    break
                end
            end
            if w != 0
                visited[w] = 1
                push!(working_stack, w)
            else
                visited[u] = 2
                push!(stack, u)
                pop!(working_stack)
            end
        end
    end
    return reverse(stack)
end

"upper triangular"
function digraph(B::AbstractMatrix)
    d = size(B, 1)
    g = DiGraph(d)
    edges = @> findall(B .!= 0) Tuple.()
    for (p, c) in edges
        add_edge!(g, p, c)
    end
    return g
end

function simulate_sem(rng::AbstractRNG, W, N, sem_type; ϵ=nothing, noise_type="gaussian", noise_scale=1.0, fault=nothing, faulty_node=nothing)  # W: upper triangular, permuted
    @assert sum(tril(W, -1)) == 0
    """X: [n_parents, N], w: [num of parents], x: [N]"""
    function _simulate_single_equation(w, x, s; ϵ)
        P = size(x, 1)  # number of parents
        P == 0 && return ϵ
        x = if sem_type == "linear"
                w'x + ϵ
            elseif sem_type == "gp"
                # gp = gaussian_process.GaussianProcessRegressor()
                # x = vec(gp.sample_y(x', random_state=rand(rng, UInt32)))' + ϵ
            elseif sem_type == "logistic"
                # rand(rng, Binomial(1, sigmoid.(w' * x)) * 1.0)
            elseif sem_type == "poisson"
                # rand(rng, Poisson(exp.(w' * x)) * 1.0)
            elseif sem_type == "mlp"
                # H = 100
                # w1 = rand(rng, Uniform(0.5, 2.0), (P, H))
                # w1[rand(rng, size(w1)...) .< 0.5] .*= -1
                # w2 = rand(rng, Uniform(0.5, 2.0), H)
                # w2[rand(rng, H) .< 0.5] .*= -1
                # w2 * s(w1 * x) + ϵ
            elseif sem_type == "mim"
                # w1 = rand(rng, Uniform(0.5, 2.0), P)
                # w1[rand(rng, P) .< 0.5] .*= -1
                # w2 = rand(rng, Uniform(0.5, 2.0), P)
                # w2[rand(rng, P) .< 0.5] .*= -1
                # w3 = rand(rng, Uniform(0.5, 2.0), P)
                # w3[rand(rng, P) .< 0.5] .*= -1
                # tanh(w1 * x) + cos(w2 * x) + sin(w3 * x) + ϵ
            elseif sem_type == "gp-add"
                # gp = GaussianProcessRegressor()
                # x = sum([gp.sample_y(x[:, i, nothing], random_state=nothing).flatten() for i in range(size(x)[1])]) + ϵ
            else
                error("unknown sem type")
            end
        return x
    end
    d = size(W, 1)
    # noise_scale = noise_scale .* ones(d)
    s = noise_scale
    if ϵ == nothing
        ϵ = noise_type ∈ ["gauss", "gaussian"] ? rand(rng, Normal(0, s), d, N) :
           noise_type ∈ ["exp", "exponential"] ? rand(rng, Exponential(s), d, N) :
                         noise_type ∈ "gumbel" ? rand(rng, Gumbel(s), d, N) :
                        noise_type ∈ "uniform" ? rand(rng, Uniform(-s, s), d, N) :
                                                 rand(rng, Normal(0, s), d, N)
    end
    sem_type = lowercase(sem_type)
    noise_type = lowercase(noise_type)
    X = zeros(d, N)
    for c in 1:d
        parents = findall(W[:, c] .!= 0)
        x = X[parents, :]
        w = W[parents, c]
        X[c, :] = _simulate_single_equation(w, x, s; ϵ=ϵ[c:c, :])
    end
    return X
end

function full_dag(top_order)
    d = length(top_order)
    A = zeros((d,d))
    for (i, var) in enumerate(top_order)
        A[var, top_order[i+1:end]] .= 1
    end
    return A
end

"""Compute various accuracy metrics for B̂.

true positive = predicted association exists in condition in correct direction
reverse = predicted association exists in condition in opposite direction
false positive = predicted association does not exist in condition

Args
    B (ndarray): [d, d] ground truth graph, {0, 1}
    B̂ (ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

Returns
    fdr: (reverse + false positive) / prediction positive
    tpr: (true positive) / condition positive
    fpr: (reverse + false positive) / condition negative
    shd: undirected extra + undirected missing + reverse
    nnz: prediction positive
"""
function dag_metrics(B̂, B; castle=false)
    castle && return metrics.MetricsDAG(B̂, B).metrics

    if any(B̂ .== -1)  # cpdag
        if !all(B̂ .∈ [-1:1])
            error("B̂ should take value in {0,1,-1}")
        end
        if any(B̂ .== -1 .&& B̂' .== -1)
            error("undirected edge should only appear once")
        end
    else  # dag
        if !all(B̂ .∈ [0:1])
            error("B̂ should take value in {0,1}")
        end
        # if !is_dag(B̂)
        #     error("B̂ should be a DAG")
        # end
    end
    d = size(B, 1)
    B′ = B'
    B̂′ = B̂'
    # @≥ B, B′, B̂, B̂′ vec.()
    # linear index of nonzeros
    pred_und = findall(vec(B̂ .== -1))
    pred = findall(vec(B̂ .== 1))
    cond = findall(vec(B .== 1))
    cond_reversed = findall(vec(B′ .== 1))
    cond_skeleton = vcat(cond, cond_reversed)
    # true pos
    true_pos = pred ∩ cond
    # treat undirected edge favorably
    true_pos_und = pred_und ∩ cond_skeleton
    true_pos = vcat(true_pos, true_pos_und)
    # false pos
    false_pos = setdiff(pred, cond_skeleton)
    false_pos_und = setdiff(pred_und, cond_skeleton)
    false_pos = vcat(false_pos, false_pos_und)
    # reverse
    extra = setdiff(pred, cond)
    reverse = extra ∩ cond_reversed
    # compute ratio
    pred_size = length(pred) + length(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - length(cond)
    fdr = (length(reverse) + length(false_pos)) / max(pred_size, 1)
    tpr = (length(true_pos)) / max(length(cond), 1)
    fpr = (length(reverse) + length(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = findall(vec(tril(B̂ + B̂′) .== 1))
    cond_lower = findall(vec(tril(B + B′) .== 1))
    extra_lower = setdiff(pred_lower, cond_lower)
    missing_lower = setdiff(cond_lower, pred_lower)
    shd = length(extra_lower) + length(missing_lower) + length(reverse)
    return (fdr = fdr, tpr = tpr, fpr = fpr, shd = shd, nnz = pred_size)
end

function plot_matrix_comparison(W, W̃, W_est_title="Estimated")
    @≥ W abs.()
    @≥ W̃ abs.()
    @> plot(heatmap(W, title="Ground truth"), heatmap(W̃, title=W_est_title), layout=(1, 2)) savefig("plots/comparisons.png")
end

function leaf_nodes(A)
    @assert sum(tril(A, -1)) == 0
    a = @> triu(A, 1) sum(dims=2) vec
    findall(a .== 0)
end

function get_ancestors(A)
    d = size(A, 2)
    C = A .+ zeros(d, d, d)
    for i = 2:d
        C[:, :, i] .= A^i
    end
    A = @> sum(C, dims=3) .> 0 dropdims(dims=3)
    leaves = leaf_nodes(A)
    ancestors = map(leaves) do l
        findall(A[:,l] .> 0)
    end
    leaves, ancestors
end

function upper(W)
    d, = size(W)
    i = map(CartesianIndex, [(i, j) for i=1:d for j=i+1:d])
    W[i]
end

function at_leat_m_ancestors(A, m)
    leaves, ancestors = get_ancestors(A)
    any(length.(ancestors) .>= m)
end

function noisy_edge_data(rng, d, e, n, an)
    # GRAPH
    B = il = leaves = ancestors = nothing
    for _=1:100
        B = simulate_dag(rng, d, e, graph_type)
        leaves, ancestors = get_ancestors(B)
        il = findfirst(length.(ancestors) .>= an)  # a leaf with ≥10 ancenstors
        il !== nothing && break
        # any(length.(ancestors) .>= 10) && break
    end
    il === nothing && error("no 10 ancestors node be found, d=$d, e=$e")
    # l = leaves[il]

    # PARAMS
    w_ranges=((0.5*20/d, 2.0*20/d), )
    W = simulate_parameters(rng, B, w_ranges)
    p = (μ = W, ρ = log(v_w))

    # OBSERVATIONS
    X = noisy_generate(rng, p, n)

    # INTERVENE
    w_range=(10, 20) .* 20 ./ d
    W̃, faulty = random_intervention(rng, W, w_range)
    Z = zero(W)
    Z[faulty...] = 1
    p̃ = (μ = W̃, ρ = log(v_w))
    X̃ = noisy_generate(rng, p̃, n)
    X

    return (; B, W, X, W̃, X̃, faulty, Z)
end

# convert networkx.digraph to DiGraph
# return graph, node labels, and indexing function by label
function from_pydigraph(ground_truth_dag, ordered_nodes)
    v = @> ordered_nodes string.()
    d = length(v)
    g = DiGraph(d)
    i_ = (node) -> findfirst(==(string(node)), v)
    for (node, children) = ground_truth_dag.graph.adjacency()
        children = keys(PyDict(children))
        i = i_(node)
        js = i_.(children)
        for j in js
            add_edge!(g, i, j);
        end
    end
    g, v, i_
end

function parent_indices(B::AbstractMatrix, j::Int)
    @assert sum(tril(B)) == 0
    findall(B[:, j] .!= 0)
end

nothing

