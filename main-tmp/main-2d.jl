include("./data2d.jl")

struct Diffusion2d{T}
    σ_max::Float32
    model::T
end
@functor Diffusion2d
# @showfields Diffusion2d
Optimisers.trainable(net::Diffusion2d) = (; net.model)

function Diffusion2d(; args)
    @assert args.input_dim == 2
    X, H, fourier_scale = (2, args.hidden_dim, args.fourier_scale)
    model = ConditionalChain(
                             Parallel(.+, Dense(2, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Dense(H, 2),
                            )
    return Diffusion2d(σ_max, model)
end

function (net::Diffusion2d)(x::AbstractMatrix{T}, t) where {T}
    h = net.model(x, t)
    σ_t = expand_dims(marginal_prob_std(t; net.σ_max), 1)
    h ./ σ_t
end

function sm_loss(net, x::AbstractMatrix{<:Real}; ϵ=1.0f-5, σ_max=25f0)
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    z = randn!(similar(x))
    σ_t = expand_dims(marginal_prob_std(t; σ_max), 1)
    x̃ = x + z .* σ_t
    score = net(x̃, t)
    return sum(abs2, score .* σ_t + z) / batchsize
end

if !isempty(args.save_path)
    net = @> Diffusion2d(; args) gpu
    args
    X = spirals_python()
    loader = Flux.DataLoader((X,) |> gpu; batchsize=32, shuffle=true);
    (x,) = @> loader first gpu
    d = size(x, 1)
    sm_loss(net, x)  # check clip_var

    opt = Flux.setup(Optimisers.Adam(args.lr_unet), net);
    loss, (grad,) = Flux.withgradient(net, ) do net
        sm_loss(net, x)
    end
    Flux.update!(opt, net, grad);

    opt = Flux.setup(Optimisers.Adam(args.lr_unet), net);
    progress = Progress(args.epochs, desc="Fitting net");
    for epoch = 1:args.epochs
        total_loss = 0.0
        for (x,) = loader
            @≥ x gpu
            global loss, (grad,) = Flux.withgradient(net, ) do net
                sm_loss(net, x)
            end
            grad
            Flux.update!(opt, net, grad)
            total_loss += loss
        end
        next!(progress; showvalues=[(:loss, total_loss/length(loader))])
    end
    # @≥ X, net cpu.();
    # BSON.@save "data/main-2d.bson" args X net

else
    BSON.@load args.load_path args X net;
    X = spirals_python()
    @≥ X, net gpu.();
end

@info "Plots"
using LaTeXStrings
xlim = ylim = (-3, 3)

#-- defaults
default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

#-- plot data
X_val = X[:, 1:100:end] |> cpu
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
xlim = ylim = (-4, 4)
x = @> Iterators.product(range(xlim..., length=50), range(ylim..., length=50)) collect vec;
x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
d = size(x, 1)
t = fill!(similar(x, size(x)[end]), 0.01) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
J = @> net(x, t)
@≥ J, x cpu.()

# x, y = eachrow(x);
# u, v = eachrow(0.2J);
# pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
# arrow0!.(x, y, u, v; as=0.2, lw=1.0);
mesh, scores = @> x', J' Array.()
plot_score_field(mesh, scores, width=0.002, vis_path="fig/score-field.png")

# @> Plots.plot(pl_data, pl_sm_data, pl_gradient; xlim, ylim, size=(1000, 800)) savefig("fig/spiral-2d.png")

