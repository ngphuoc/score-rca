using Flux, CUDA, MLUtils, EndpointRanges
include("lib/utils.jl")
include("dsm-model.jl")

@info "Step1: Train diffusion model"

@≥ z, x, l, s, z3, x3, l3, s3, za, xa, la, sa gpu.()
d = size(z, 1)

H = args.hidden_dim
σ_max = 4std(za)  # args.σ_max
fourier_scale = 2σ_max  # args.fourier_scale
net = ConditionalChain(
                 Parallel(.+, Dense(d, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Dense(H, d),
                )
diffusion_model = @> DSM(args.σ_max, net) gpu

mpath = replace(fpath, "/data-" => "/model-")
if !isfile(mpath)
    diffusion_model = train_dsm(diffusion_model, z; args)
    @≥ diffusion_model, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa cpu.()
    @info "Saving diffusion model to $mpath"
    BSON.@save mpath args diffusion_model g z x l s z3 x3 l3 s3 za xa la sa
end
@info "Loading diffusion model from $mpath"
BSON.@load mpath args diffusion_model g z x l s z3 x3 l3 s3 za xa la sa
@≥ diffusion_model, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa gpu.()

@info "Step2: Sampling k in-distribution points Xs and k diffusion trajectories X_{t} by inversing the noise z_{j} and reverse diffusion from z_{j}"

function reverse_diffusion(diffusion_model, init_x; n_steps = 100, σ_max = args.σ_max, ϵ = 1f-3)
    time_steps = @> LinRange(1.0f0, ϵ, n_steps) collect
    Δt = time_steps[1] - time_steps[2]
    xs, ∇xs, ms, ∇ms = Euler_Maruyama_sampler(diffusion_model, init_x, time_steps, Δt, σ_max)
    return xs, ∇xs, ms, ∇ms
end

adjmat = @> g.dag adjacency_matrix Matrix{Bool}
ii = @> adjmat eachcol findall.()  # use global indices to avoid mutation error in autograd
g.cpds = gpu.(g.cpds)
forwardg(x) = forward(g, x)

init_z = z
n_steps = 10
zs, ∇zs, ms, ∇ms = reverse_diffusion(diffusion_model, init_z; n_steps) |> cpu
xs, ss = @> zs gpu reshape(d, :) forwardg reshape.(Ref(size(zs)))
@≥ init_z cpu;
# Δz for Eq. 14
Δzs = @> cat(init_z, zs, dims=3) diff(dims=3)

∇f, = Zygote.gradient(gpu(reshape(zs, size(zs, 1), :)), ) do z
    @> forward_leaf(g, z, ii) sum
end;
@≥ ∇f cpu reshape((d, :, n_steps));

@info "Step3: Compute the score using Eq. 14"

# Compute the scores Eq. 14:
# ξ_j(z,z′) ≈ (z_j − z_j′) Σ − s_nj(z(tk); θ) (z_j(t_i) − z_j(t_i − 1))
# - (x-x') \int ∂f_i/∂_x_j * dx

function get_scores(∇ms, ∇f, zs, Δzs)
    scores = -∇ms .* ∇f .* Δzs
    full_path = @> scores sum(dims=3) squeezeall
    half_path = @> scores[:, :, 1:end÷2] sum(dims=3) squeezeall
    full_path, half_path
end

@info "Step4: Get ground truth rankings"

max_k = args.n_anomaly_nodes
overall_max_k = max_k + 1

function get_gt_z_ranking(za, ∇za)
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
    @≥ scores hcats
    return scores
end

∇za, = Zygote.gradient(za, ) do za
    @> forward_leaf(g, za, ii) sum
end
@≥ za, ∇za cpu
gt_ranking = get_gt_z_ranking(za, ∇za)
# gt_ranking = get_gt_z_ranking(za[:, :, 1], ∇za[:, :, end÷2])
@> gt_ranking mean(dims=2)
gt_manual = indexin(1:d, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))

@info "Step5: Outlier ndcg ranking"

@≥ za gpu
init_z = za
n_steps = 10
za, ∇za, ma, ∇ma = reverse_diffusion(diffusion_model, init_z; n_steps) |> cpu
xa, sa = @> za gpu reshape(d, :) forwardg reshape.(Ref(size(za)))
@≥ init_z cpu;
Δza = @> cat(init_z, za, dims=3) diff(dims=3)  # Δz for Eq. 14
∇fa, = Zygote.gradient(gpu(reshape(za, size(za, 1), :)), ) do z
    @> forward_leaf(g, z, ii) sum
end;
@≥ ∇fa cpu reshape((d, :, n_steps));

anomaly_measure_full, anomaly_measure_half = get_scores(∇ma, ∇fa, za, Δza)

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

@info "Step6: Save results"

k = 2
res = map(1:d) do k
    results = (;
                 map(round3 ∘ only ∘ PyArray, (;
                     ranking_full = ndcg_score(gt_ranking', anomaly_measure_full'; k),
                     manual_full = ndcg_score(gt_manual', anomaly_measure_full'; k),
                     ranking_half = ndcg_score(gt_ranking', anomaly_measure_half'; k),
                     manual_half = ndcg_score(gt_manual', anomaly_measure_half'; k),
                 ))...,
                 k, args.data_id, args.n_nodes, args.n_anomaly_nodes, method = "SIREN", fpath,
              )
end
@> DataFrame(res) println

append!(results, res)

