# TODO:
# v Train score functions separately for each node
# v Compute sensitivities wrt each node
# ? Inverse noises
# - Compute rca score by combining scores and sensitivities

using Flux, CUDA, MLUtils, EndpointRanges
include("data-rca.jl")
include("lib/utils.jl")
include("dsm-model.jl")

g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, fpath = load_data(args; dir="datals");
d = size(x, 1)
g.cpds[1]
g.cpds[2]

@info "# Inputs Score function estimators s_{j}^{i}, mean functions f_{i}, observation X with outlier observed at the leaf X_{n}"

@assert x ≈ l + s .* z
z = (x - l) ./ s
@≥ z, x, l, s, z3, x3, l3, s3, za, xa, la, sa gpu.()
d = size(z, 1)

# Train score function on data with mean removed"

H, fourier_scale = args.hidden_dim, args.fourier_scale
net = ConditionalChain(
                 Parallel(.+, Dense(d, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                 Dense(H, d),
                )
diffusion_model = @> DSM(args.σ_max, net) gpu

mpath = replace(fpath, ".bson" => "-model.bson")
if !isfile(mpath)
    @info "Step0: Training diffusion model"
    diffusion_model = train_dsm(diffusion_model, z; args)
    @≥ diffusion_model, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa cpu.()
    @info "Step0: Saving diffusion model to $mpath"
    BSON.@save mpath args diffusion_model g z x l s z3 x3 l3 s3 za xa la sa
end
BSON.@load mpath args diffusion_model g z x l s z3 x3 l3 s3 za xa la sa
@≥ diffusion_model, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa gpu.()

@info "Step1: Sampling k in-distribution points Xs and k diffusion trajectories X_{t} by inversing the noise z_{j} and reverse diffusion from z_{j}"

function langevine_ref(diffusion_model, init_x; n_steps = 100, σ_max = args.σ_max, ϵ = 1f-3)
    time_steps = @> LinRange(1.0f0, ϵ, n_steps) collect
    Δt = time_steps[1] - time_steps[2]
    xs, ms, ∇xs = Euler_Maruyama_sampler(diffusion_model, init_x, time_steps, Δt, σ_max)
    return xs, ms, ∇xs
end

forwardg(x) = forward(g, x)

init_z = z
zs, ms, ∇zs = langevine_ref(diffusion_model, init_z, n_steps = 100) |> cpu
xs, ss = @> zs reshape(d, :) forwardg reshape.(Ref(size(zs)))

@info "Step2: Forward using the mean functions f_{i} and backprop to calculate the sensitivities"

adjmat = @> g.dag adjacency_matrix Matrix{Bool}
ii = @> adjmat eachcol findall.()  # use global indices to avoid mutation error in autograd
size_zs = size(zs)
z = @> zs reshape(d, :)
@> forward_leaf(g, z, ii) sum

∇f, = Zygote.gradient(z, ) do z
    @> forward_leaf(g, z, ii) sum
end

# @info "Step3: Compute the score using Eq. 18-20"

# jac = zero(adjmat)

# max_k = args.n_anomaly_nodes
# overall_max_k = max_k + 1
# adjmat = @> g.dag adjacency_matrix Matrix{Bool}
# d = size(adjmat, 1)
# ii = @> adjmat eachcol findall.()
# function get_z_rankings(za, ∇za)
#     @assert size(za, 1) == d
#     i = 1
#     scores = Vector{Float64}[]  # 1 score vector for each outlier
#     batchsize = size(za, 2)
#     for i = 1:batchsize
#         tmp = Dict(j => ∇za[j, i] * za[j, i] for j = 1:d)
#         ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]   # used
#         score = zeros(d)
#         for q in 1:max_k
#             iq = findfirst(==(ranking[q]), 1:d)
#             score[iq] = overall_max_k - q
#         end
#         push!(scores, score)
#     end
#     return scores
# end

