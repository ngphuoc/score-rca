using Revise
using Flux
include("./data-rca.jl")
include("./lib/utils.jl")
include("./denoising-score-matching.jl")
include("./plot-dsm.jl")

const to_device = args.to_device

g, x, x′, xa, y, y′, ya, ε, ε′, εa, μx, σx, anomaly_nodes = load_normalised_data(args);

@info "#-- 1. collapse data"


@assert x ≈ y + ε
z = x - y
@≥ z vec transpose;
@≥ z, x, x′, xa, y, y′, ya, ε, ε′, εa, μx, σx to_device.()

@info "#-- 2. train score function on data with mean removed"

H, fourier_scale = args.hidden_dim, args.fourier_scale
net = ConditionalChain(
                 Parallel(.+, Dense(1, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Dense(H, 1),
                )
dnet = @> DSM(args.σ_max, net) to_device
dnet = train_dsm(dnet, z; args)

function get_score(dnet, x)
    t = fill!(similar(x, size(x)[end]), 0.01) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    # σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    @> dnet(x, t)
end

get_scores(dnet, x) = @> get_score.([dnet], transpose.(eachrow(x))) vcats

dz = get_score(dnet, z)
dx = get_scores(dnet, x)

@info "#-- 3. reference points and outlier scores"


function get_ref(x, r)
    dist_xr = pairwise(Euclidean(), x, r)  # correct
    _, j = @> findmin(dist_xr, dims=2)
    @≥ j getindex.(2) vec
    r[:, j]
end

r = get_ref(xa, x)
dx = get_scores(dnet, x)
dr = get_scores(dnet, r)

vx = abs.(dx)  # anomaly_measure

@info "#-- 4. ground truth ranking and results"

max_k = args.n_anomaly_nodes
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

μa = forward_1step_scaled(g, xa, μx, σx)
ε̂a = xa - μa
xr = get_ref(xa, x);  # reference points
μr = forward_1step_scaled(g, xr, μx, σx)
ε̂r = xr - μr

@≥ ε̂a, ε̂r to_device
∇xa = @> get_scores(dnet, ε̂a)
anomaly_measure = abs.(∇xa)

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

gt_manual = indexin(1:args.n_nodes, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))
ndcg_score(gt_manual', abs.((ε̂a - ε̂r) .* ∇xa)', k=args.n_anomaly_nodes)

@info "#-- 5. save results"

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
    ndcg_ranking = ndcg_score(gt_value', abs.((ε̂a - ε̂r) .* ∇xa)'; k)
    ndcg_manual = ndcg_score(gt_manual', abs.((ε̂a - ε̂r) .* ∇xa)'; k)
    @≥ ndcg_ranking, ndcg_manual PyArray.() only.()
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "DSM", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, k])
end

println(df);

fname = "results/random-graphs.csv"
CSV.write(fname, df, header=!isfile(fname), append=true)

