include("./data2d.jl")
@info "Model"

struct Diffusion2d{T}
    σ_max::Float32
    model::T
end

function RoundTimesteps(max_timesteps::Int)
    function round_timesteps(t::AbstractArray{T,N}) where {T<:Real,N}
        round.(Int, max_timesteps .* t)
        # round.(max_timesteps .* t)
    end
end

@functor Diffusion2d
# @showfields Diffusion2d
Optimisers.trainable(unet::Diffusion2d) = (; unet.model)

function Diffusion2d(; args)
    @assert args.input_dim == 2
    X, H, E, n_timesteps = (2, args.hidden_dim, args.embed_dim, args.n_timesteps)
    model = ConditionalChain(
                             Parallel(.+, Dense(2, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Dense(H, 2),
                            )
    return Diffusion2d(σ_max, model)
end

function (unet::Diffusion2d)(x::AbstractMatrix{T}, t) where {T}
    h = unet.model(x, t)
    σ_t = expand_dims(marginal_prob_std(t; unet.σ_max), 1)
    h ./ σ_t
end

unet = @> Diffusion2d(; args) gpu
score_matching_loss(unet, x)  # check clip_var

opt = Flux.setup(Optimisers.Adam(args.lr_unet), unet);
loss, (grad,) = Flux.withgradient(unet, ) do unet
    score_matching_loss(unet, x)
end
Flux.update!(opt, unet, grad);

opt = Flux.setup(Optimisers.Adam(args.lr_unet), unet);
progress = Progress(args.epochs, desc="Fitting unet");
for epoch = 1:args.epochs
    total_loss = 0.0
    for (x,) = loader
        @≥ x gpu
        loss, (grad,) = Flux.withgradient(unet, ) do unet
            score_matching_loss(unet, x)
        end
        grad
        Flux.update!(opt, unet, grad)
        total_loss += loss
    end
    next!(progress; showvalues=[(:loss, total_loss/length(loader))])
end

@info "Plots"
using LaTeXStrings
include("lib/plot-utils.jl")
xlim = ylim = (-5, 5)

#-- defaults
default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

#-- plot data
x, y = eachrow(X_val)
pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

#-- plot perturbed data
x = @> X_val gpu;
d = size(x, 1)
X, batchsize = size(x, 1), size(x)[end]
t = rand!(similar(x, size(x)[end])) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
z = 2rand!(similar(x)) .- 1;
x̃ = x .+ σ_t .* z
x, y = eachrow(cpu(x̃))
pl_sm_data = scatter(x, y; xlab=L"x", ylab=L"y", title="Perturbed score matching")

#-- plot gradients
μx, σx = @> X_val mean(dims=2), std(dims=2)
xlim = ylim = (-2, 2)
x = @> Iterators.product(range(xlim..., length=20), range(ylim..., length=20)) collect vec;
x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
d = size(x, 1)
t = fill!(similar(x, size(x)[end]), 0.1) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
J = @> unet(x, t)
@≥ J, x cpu.()
x, y = eachrow(x);
u, v = eachrow(0.2J);
pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
arrow0!.(x, y, u, v; as=0.2, lw=1.0);

@> Plots.plot(pl_data, pl_sm_data, pl_gradient; xlim, ylim, size=(1000, 800)) savefig("fig/spiral-2d.png")

