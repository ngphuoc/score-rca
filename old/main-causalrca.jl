using Flux, ShapML
using Flux: Data
include("./data-rca.jl")
include("./lib/utils.jl")
include("./denoising-score-matching.jl")
include("./plot-dsm.jl")
seed = 1

const to_device = args.to_device

g, x, x3, xa, y, y3, ya, ε, ε3, εa, μx, σx, anomaly_nodes = load_normalised_data(args);

@info "#-- 1. fit linear bayesnet"

@assert x ≈ y + ε
z = x - y
@≥ z vec transpose;
@≥ z, x, x3, xa, y, y3, ya, ε, ε3, εa, μx, σx to_device.()

bn = copy_linear_dag(g)
g.cpds
bn.cpds
Distributions.fit!(bn, x)

@info "#-- 2. define residual function as outlier scores"

function predict_function(bn, noise)
    DataFrame(ŷ = forward_leaf(bn, Array(noise)'))
end

function inverse_noise(bn, x)
    x - forward_1step(bn, x)
end

ε̂a = inverse_noise(bn, xa)
explain = DataFrame(ε̂a', names(g))
ε̂ = inverse_noise(bn, x)
reference = DataFrame(ε̂', names(g))

sample_size = 50  # Number of Monte Carlo samples.
data_shap = ShapML.shap(; explain , reference , model=bn , predict_function , sample_size , seed ,)
show(data_shap, allcols = true)
m = nrow(explain)
a = @> data_shap[!, :feature_name] reshape(m, :)
anomaly_measure = @> data_shap[!, :shap_effect] reshape(m, :)
anomaly_nodes

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

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

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

k = 1
for k=1:args.min_depth
    ndcg_ranking = ndcg_score(gt_value', anomaly_measure; k)
    ndcg_manual = ndcg_score(gt_manual', anomaly_measure; k)
    @≥ ndcg_ranking, ndcg_manual PyArray.() only.()
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "Shapley", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, k])
end

println(df);

fname = "results/random-graphs.csv"
CSV.write(fname, df, header=!isfile(fname), append=true)
