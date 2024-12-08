using Revise
using Flux
include("lib/utils.jl")
include("denoising-score-matching.jl")
include("plot-dsm.jl")
include("data-rca.jl")

const to_device = args.to_device

g, x, x3, xa, y, y3, ya, ε, ε3, εa, μx, σx, anomaly_nodes = load_normalised_data(args);
B = @> g.dag adjacency_matrix Matrix{Bool}
anomaly_nodes

@info "#-- 1. fit linear bayesnet"

@assert x ≈ y + ε
@≥ x, x3, xa, y, y3, ya, ε, ε3, εa, μx, σx to_device.()

@info "#-- 2. define residual function as outlier scores"

@info "#-- 3. ground truth ranking and results"

max_k = n_anomaly_nodes
overall_max_k = max_k + 1
adjmat = @> g.dag adjacency_matrix Matrix{Bool}
d = size(adjmat, 1)
ii = @> adjmat eachcol findall.()
function get_ε_rankings(εa, ∇εa)
    @assert size(εa, 1) == d
    i = 1
    scores = Vector{Float64}[]  # 1 score vector for each outlier
    batchsize = size(εa, 2)
    for i = 1:batchsize
        tmp = Dict(j => ∇εa[j, i] * εa[j, i] for j = 1:d)
        ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]  # used
        score = zeros(d)
        for q in 1:max_k
            iq = findfirst(==(ranking[q]), 1:d)
            score[iq] = overall_max_k - q
        end
        push!(scores, score)
    end
    return scores
end

# TODO: recheck gt ranking scores
∇εa, = Zygote.gradient(εa, ) do εa
    @> forward_leaf(g, εa, ii) sum
end
gt_value = @> get_ε_rankings(εa, ∇εa) hcats
@> gt_value mean(dims=2)
anomaly_nodes

function zscore(x, r)
    μ, σ = @> r mean(dims=2), std(dims=2)
    @. (x - μ) / σ
end

""" like matmul but reduce by maximum/minimum/f instead of sum
"""
function fmul(f, A::AbstractMatrix{Bool}, x1)
    d = size(A, 1)
    y = similar(x1)
    for j = 1:d
        a = A[j, :]
        z = (a .* x1)[a]
        y[j] = length(z) > 0 ? f(z) : 0f0
    end
    y
end

maxmul(A, x1) = fmul(maximum, A, x1)
minmul(A, x1) = fmul(minimum, A, x1)

""" Subgraph connecting node j and node l"""
function path_graph(dag, j, l)
    g = DiGraph(nv(dag))
    for e = a_star(dag, j, l)
        add_edge!(g, e.src, e.dst);
    end
    g
end

function traversal_measure(g, xa, x)
    d = nv(g.dag)
    v = Distributions.cdf(Normal(0, 1), -abs.(zscore(xa, x)))
    B = @> adjacency_matrix(g.dag) Matrix{Bool}
    A = @> I(d) + B Matrix{Bool}
    x1 = v[:, 2]
    ys = []
    for x1 = eachcol(v)
        # traversal algorithm - no abnormal before j, all abnormal after j: minscore(>=j) - maxscore(<j)
        y = similar(x1)
        j = 2
        @≥ x1 Array
        for j = 1:d
            a1, b1 = @> copy(x1), copy(x1) Array{Float32}.()
            a1[j] = 0
            as = [a1]  # before j
            # no abnormal before j: propagate max from root to j
            for i = 1:j-1
            # for i = 1:1
                xj = copy(as[end])
                push!(as, maxmul(A', xj))
            end

            # all abnormal after j: propagate min from j to leaf d, path at most length j+1:d
            P = @> I(d) + adjacency_matrix(path_graph(g.dag, j, d)) Matrix{Bool}
            bs = [b1]  # after j
            for i = j+1:d
            # for i = 1:1
                push!(bs, minmul(P', bs[end]))
            end
            # minscore(>=j) - maxscore(<j)
            y[j] = bs[end][d] - as[end][j]
        end
        hcat(x1, y)
        A
        push!(ys, y)
    end
    hcats(ys)
end

function naive_measure(g, xa, x)
    d = nv(g.dag)
    v = Distributions.cdf(Normal(0, 1), -abs.(zscore(xa, x)))
end

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

d
gt_manual = indexin(1:d, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))

@info "#-- 4. save results"

df = DataFrame(
               n_nodes = Int[],
               n_anomaly_nodes = Int[],
               method = String[],
               noise_dist  = String[],
               data_id = Int[],
               ndcg_ranking = Float64[],
               ndcg_manual = Float64[],
               k = Int[],
              )

anomaly_measure = traversal_measure(g, xa, x)
k = 1
for k=1:d-1
    ndcg_ranking = ndcg_score(gt_value', anomaly_measure'; k)
    ndcg_manual = ndcg_score(gt_manual', anomaly_measure'; k)
    @≥ ndcg_ranking, ndcg_manual PyArray.() only.()
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "traversal", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, k])
end

anomaly_measure = naive_measure(g, xa, x)
k = 1
for k=1:args.min_depth
    ndcg_ranking = ndcg_score(gt_value', anomaly_measure'; k)
    ndcg_manual = ndcg_score(gt_manual', anomaly_measure'; k)
    @≥ ndcg_ranking, ndcg_manual PyArray.() only.()
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "naive", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, k])
end

println(df);

fname = "results/random-graphs.csv"
CSV.write(fname, df, header=!isfile(fname), append=true)

