@info "#-- 1. fit linear bayesnet"

@≥ x, xa, ε, εa args.to_device.()
bn = copy_bayesian_dag(g)
g.cpds
bn.cpds
Distributions.fit!(bn, x)

@info "#-- 2. define residual function as outlier scores"

μx = forward_1step(bn, x)

function get_residual(bn, x)
    μx = forward_1step(bn, x)
    x - μx
end

ε̂x = get_residual(bn, x)
vx = abs.(ε̂x)  # anomaly_measure

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

μa = forward_1step(g, xa)
ε̂a = xa - μa
anomaly_measure = abs.(get_residual(bn, xa))  # anomaly_measure

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

gt_manual = indexin(1:d, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))

@info "#-- 4. save results"

df = copy(dfs[1:0, :])

k = 1
for k=1:d-1
    ndcg_ranking = ndcg_score(gt_value', anomaly_measure'; k)
    ndcg_manual = ndcg_score(gt_manual', anomaly_measure'; k)
    @≥ ndcg_ranking, ndcg_manual PyArray.() only.()
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "BIGEN", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, k])
end

println(df);

append!(dfs, df)
