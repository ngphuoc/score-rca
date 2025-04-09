# TODO:
# v Train score functions separately for each node
# v Compute sensitivities wrt each node
# ? Inverse noises
# - Compute rca score by combining scores and sensitivities

using Flux, CUDA, MLUtils, EndpointRanges
include("data-rca.jl")
include("lib/utils.jl")
include("dsm-model.jl")

to_device = args.to_device
g, x, x3, xa, f, f3, fa, z, z3, za, anomaly_nodes = load_data(args; dir="datals");
d = size(x, 1)
g.cpds[1]
g.cpds[2]

@info "# Inputs Score function estimators s_{j}^{i}, mean functions f_{i}, observation X with outlier observed at the leaf X_{n}"

@assert x ≈ f + z
z = x - f
@≥ z, x, x3, xa, f, f3, fa, z, z3, za, to_device.()
d = size(z, 1)

# Train score function on data with mean removed"

H, fourier_scale = args.hidden_dim, args.fourier_scale
net = ConditionalChain(
                 Parallel(.+, Dense(d, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Dense(H, d),
                )
diffusion_model = @> DSM(args.σ_max, net) to_device

if args.training
    diffusion_model = train_dsm(diffusion_model, z; args)
    @≥ diffusion_model, z, x, x3, xa, f, f3, fa, z, z3, za cpu.()
    BSON.@save "data/main-siren.bson" args diffusion_model z x x3 xa f f3 fa z z3 za
end

BSON.@load "data/main-siren.bson" diffusion_model z x x3 xa f f3 fa z z3 za
@≥ diffusion_model, z, x, x3, xa, f, f3, fa, z, z3, za gpu.()

@info "Step1: Sampling k in-distribution points Xs and k diffusion trajectories X_{t} by inversing the noise z_{j} and reverse diffusion from z_{j}"

function langevine_ref(diffusion_model, init_x; n_steps = 100, σ_max = args.σ_max, ϵ = 1f-3)
    time_steps = @> LinRange(1.0f0, ϵ, n_steps) collect
    Δt = time_steps[1] - time_steps[2]
    xs, ms, ∇xs = Euler_Maruyama_sampler(diffusion_model, init_x, time_steps, Δt, σ_max)
    return xs, ms, ∇xs
end

forwardg(x) = forward(g, x)

@info "Step2: Forward using the mean functions f_{i} and backprop to calculate the sensitivities"

adjmat = @> g.dag adjacency_matrix Matrix{Bool}
ii = @> adjmat eachcol findall.()  # use global indices to avoid mutation error in autograd
@≥ zs reshape(d, :) cpu
xs = forward(g, zs)
∇fn, = Zygote.gradient(zs, ) do zs
    @> forward_leaf(g, zs, ii) sum
end

@info "Step3: Compute the score using Eq. 18-20"

jac = zero(adjmat)

max_k = args.n_anomaly_nodes
overall_max_k = max_k + 1
adjmat = @> g.dag adjacency_matrix Matrix{Bool}
d = size(adjmat, 1)
ii = @> adjmat eachcol findall.()
function get_z_rankings(za, ∇za)
    @assert size(za, 1) == d
    i = 1
    scores = Vector{Float64}[]  # 1 score vector for each outlier
    batchsize = size(za, 2)
    for i = 1:batchsize
        tmp = Dict(j => ∇za[j, i] * za[j, i] for j = 1:d)
        ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]   # used
        score = zeros(d)
        for q in 1:max_k
            iq = findfirst(==(ranking[q]), 1:d)
            score[iq] = overall_max_k - q
        end
        push!(scores, score)
    end
    return scores
end

@info "Step4: Integrate scores along the trajectories and multiply by the distance \frac{1}{2}(x_{j}-x'_{j}) for each j using Eq. 16"

# TODO: recheck gt ranking scores
∇za, = Zygote.gradient(za, ) do za
    @> forward_leaf(g, za, ii) sum
end
gt_value = @> get_z_rankings(za, ∇za) hcats
@> gt_value mean(dims=2)
anomaly_nodes

@≥ ẑa, ẑr to_device
∇xa = @> get_scores(diffusion_model, ẑa)
anomaly_measure = abs.(∇xa)

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

gt_manual = indexin(1:d, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))
ndcg_score(gt_manual', abs.((ẑa - ẑr) .* ∇xa)', k=args.n_anomaly_nodes)

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
anomaly_measure = abs.((ẑa - ẑr) .* ∇xa)'
# a = anomaly_measure
# @≥ a reshape(:, 5, 4)
# i = @> a std(dims=3) squeeze(3) argmax(dims=2) vec
# a[i, :]
for k=1:args.min_depth
    ndcg_ranking = ndcg_score(gt_value', anomaly_measure; k)
    ndcg_manual = ndcg_score(gt_manual', anomaly_measure; k)
    @≥ ndcg_ranking, ndcg_manual PyArray.() only.()
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "DSM", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, k])
end

println(df);

fname = "results/random-graphs.csv"
# CSV.write(fname, df, header=!isfile(fname), append=true)

