include("denoising-score-matching.jl")
include("plot-dsm.jl")
include("lib/diffusion.jl")

@info "#-- 1. collapse data"

# multiple siren models with different hyperparams
# hidden size
# fourier_scale
# sigma
# visualize 2d gradient field

z = ε
@> ε mean(dims=2)
@> ε std(dims=2)

@≥ z vec transpose;
@≥ z, x, xa, ε, εa args.to_device.()

@info "#-- 2. train score function on data with mean removed"

σ_max = 2std(εa)
fourier_scale = 2σ_max
hidden_dim = 50
dnet = train_dsm(DSM(
                     σ_max,
                     ConditionalChain(
                                      Parallel(.+, Dense(1, hidden_dim), Chain(RandomFourierFeatures(hidden_dim, fourier_scale), Dense(hidden_dim, hidden_dim))), swish,
                                      Dense(hidden_dim, 1),
                                     )
                    ), z; args)

# TODO: mixture of scales
function get_score(dnet, x)
    t = 0.01ones_like(x) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    # σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    @> dnet(x, t)
end

get_scores(dnet, x) = @> get_score.([dnet], transpose.(eachrow(x))) vcats

@info "#-- 3. reference points and outlier scores"

function get_ref(x, r)
    dist_xr = pairwise(Euclidean(), x, r)  # correct
    _, j = @> findmin(dist_xr, dims=2)
    @≥ j getindex.(2) vec
    r[:, j]
end

# function setup_sampler(; reverse_steps=100, ϵ=1.0f-3)
#     time_steps = LinRange(1.0f0, ϵ, reverse_steps)
#     Δt = time_steps[1] - time_steps[2]
#     return time_steps, Δt
# end

function langevine_ref(dnet, init_x; reverse_steps=100, ϵ=1.0f-3)
    time_steps = @> LinRange(1.0f0, ϵ, reverse_steps) collect
    Δt = time_steps[1] - time_steps[2]
    time_steps .= 1f-3
    r, ∇s = Euler_Maruyama_sampler(dnet, init_x, time_steps, Δt, σ_max)
    @≥ r reshape(size(εa))
    @≥ ∇s reshape(size(εa)..., :)
    return r, ∇s
end

@info "#-- 4. ground truth ranking and results"
# to multiply with ya
# raw ya vs. scaling with gradient

max_k = n_anomaly_nodes
overall_max_k = max_k + 1
adjmat = @> g.dag adjacency_matrix Matrix{Bool}
d = size(adjmat, 1)
ii = @> adjmat eachcol findall.()

# TODO: recheck gt ranking scores
∇εa, = Zygote.gradient(εa, ) do εa
    @> forward_leaf(g, εa, ii) sum
end

gt_value = @> get_ε_rankings(ya, ∇εa) hcats
gt_pvalue = @> ya .* ∇εa abs.()
@> gt_value mean(dims=2)
anomaly_nodes

# μa = forward_1step_scaled(g, xa, μx, σx)
# ε̂a = xa - μa
# xr = get_ref(xa, x);  # reference points
# μr = forward_1step_scaled(g, xr, μx, σx)
# ε̂r = xr - μr
ε̂a = εa
init_ε = @> εa vec transpose Array
ε̂r, ∇s = langevine_ref(dnet, init_ε)

# autograd score

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

gt_manual = indexin(1:d, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))
# ndcg_score(gt_manual', abs.((ε̂a - ε̂r) .* ∇xa)', k=args.n_anomaly_nodes)

@info "#-- 5. save results"

function siren_value(ε̂a, ε̂r, ∇s)
    T = size(∇s)[end]
    @> abs.(mean(∇s, dims=3) .* (ε̂a - ε̂r)) squeeze(3)
end

function siren_value2(ε̂a, ε̂r, ∇s)
    T = size(∇s)[end]
    @> abs.(mean(∇s[:, :, [1, T]], dims=3) .* (ε̂a - ε̂r)) squeeze(3)
end

df = copy(dfs[1:0, :])

k = 1
# anomaly_value = abs.((∇a + ∇r) .* (ε̂a - ε̂r))
anomaly_value = siren_value(ε̂a, ε̂r, ∇s)
if d == 1
    @≥ gt_value, gt_manual, gt_pvalue, anomaly_value repeat.(outer=(2, 1))
end
for k=1:d
    ndcg_ranking = ndcg_score(gt_value', anomaly_value'; k)
    ndcg_manual = ndcg_score(gt_manual', anomaly_value'; k)
    ndcg_pvalue = ndcg_score(gt_pvalue', anomaly_value'; k)
    @≥ ndcg_ranking, ndcg_manual, ndcg_pvalue PyArray.() only.() round.(digits=3)
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "SIREN", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, ndcg_pvalue, k])
end

println(df);

append!(dfs, df)

