include("./data-rca.jl")

struct DSM{T}
    σ_max::Float32
    model::T
end
@functor DSM
# @showfields DSM
Optimisers.trainable(dnet::DSM) = (; dnet.model)

function DSM(n_groups; args)
    X, H, F, fourier_scale = (args.input_dim, args.hidden_dim, n_groups, args.fourier_scale)
    model = ConditionalChain(
                             Parallel(.+, Dense(X, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
                             Dense(H, X),
                            )
    return DSM(σ_max, model)
end

function (dnet::DSM)(x::AbstractMatrix{T}, t) where {T}
    h = dnet.model(x, t)
    σ_t = expand_dims(marginal_prob_std(t; dnet.σ_max), 1)
    h ./ σ_t
end

function dsm_loss(dnet, x::AbstractMatrix{<:Real}; ϵ=1.0f-5, σ_max)
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1f0 - ϵ) .+ ϵ  # same t for j and paj
    z = randn!(similar(x))
    σ_t = expand_dims(marginal_prob_std(t; σ_max), 1)
    x̃ = x + z .* σ_t
    score = dnet(x̃, t)
    return sum(abs2, score .* σ_t + z) / batchsize
end

@info "Create graph and data"
n_nodes = round(Int, min_depth^2 / 3 + 1)
n_root_nodes = 1
n_downstream_nodes = n_nodes - n_root_nodes
scale, hidden = 0.5, 100
dag = random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, scale, hidden)
normal_df, perturbed_df, anomaly_df = draw_normal_perturbed_anomaly(dag; args)

@info "Normalise data"
X = @> vcat(normal_df, perturbed_df) Array transpose Array;
inputdim = size(X, 1)
μX, σX = @> X mean(dims=2), std(dims=2);
normal_df = @. (normal_df - μX') / σX'
perturbed_df = @. (perturbed_df - μX') / σX'
anomaly_df = @. (anomaly_df - μX') / σX'

@show @> normal_df Array minimum(dims=1), maximum(dims=1)
@show @> perturbed_df Array minimum(dims=1), maximum(dims=1)

fig_path = "fig/3data-2d.png"
@info "Plot normal, perturbed, anomaly data ($fig_path)"
xlim = ylim = (-3, 3)
plot_3data(normal_df, perturbed_df, anomaly_df; xlim, ylim, fig_path)

@info "Training dsm net"
ϵ = 1.0f-5
dnet = @> DSM(inputdim; args) gpu
# dnet = train_dsm(dnet, normal_df; args)
df = vcat(normal_df, perturbed_df)
X = @> df Array;
loader = DataLoader((X',); args.batchsize, shuffle=true)
(x,) = @> loader first gpu
d = size(x, 1)
batchsize = size(x)[end]
t = rand!(similar(x, batchsize)) .* (1f0 - ϵ) .+ ϵ  # same t for j and paj
dnet(x, t)
dsm_loss(dnet, x; args.σ_max)

# eval_unet(dnet, df)
opt = Flux.setup(Optimisers.Adam(args.lr), dnet);
progress = Progress(args.epochs, desc="Fitting dnet")
for epoch = 1:args.epochs
    total_loss = 0.0
    for (x,) = loader
        @≥ x gpu
        global loss, (grad,) = Flux.withgradient(dnet, ) do dnet
            dsm_loss(dnet, x; args.σ_max)
        end
        Flux.update!(opt, dnet, grad)
        total_loss += loss
    end
    next!(progress; showvalues=[(:loss, total_loss/length(loader))])
end

# @≥ X, dnet cpu.();
# BSON.@save "data/main-rca.bson" args X dnet
BSON.@load "data/main-rca.bson" args X dnet
@≥ X, dnet gpu.();

@info "Plots"
using LaTeXStrings
xlim = ylim = (-3, 3)

#-- defaults
default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

dsm_loss(dnet, x; args.σ_max)
X_val = X[1:100:end, :]' |> cpu
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
J = @> dnet(x, t)
@≥ J, x cpu.()

# x, y = eachrow(x);
# u, v = eachrow(0.2J);
# pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
# arrow0!.(x, y, u, v; as=0.2, lw=1.0);
mesh, scores = @> x', J' Array.()
plot_score_field(mesh, scores, width=0.002, vis_path="fig/score-field.png")

@> Plots.plot(pl_data, pl_sm_data; xlim, ylim, size=(1000, 800)) savefig("fig/main-rca-2d.png")

