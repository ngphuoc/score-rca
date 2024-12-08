using MultivariateStats, Distributions, StatsPlots, Plots, LaTeXStrings, Plots.PlotMeasures
using Plots: plot
import Base.show, Base.eltype
import Flux._big_show, Flux._show_children
import NNlib: batched_mul
using Revise
using BSON, JSON, DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2, Dates, Flux, Optimisers, Plots, Printf, ProgressMeter, Random, Distances
using Flux.Data: DataLoader
using Flux: crossentropy
using Optimisers: Optimisers, trainable
using ColorSchemes
color_pallete = Plots.palette(:Paired_10)
colors = color_pallete.colors

# include("./imports.jl")
include("./lib/utils.jl")
include("./lib/graph.jl")
include("./lib/outlier.jl")
include("./lib/diffusion.jl")
include("./lib/graph.jl")
include("./lib/nn.jl")
include("./lib/nnlib.jl")
include("./lib/utils.jl")
include("./mlp-unet-2d.jl")
# include("./random-graph-datasets.jl")
include("./rca.jl")
include("utilities.jl")
include("models/embed.jl")
include("models/ConditionalChain.jl")
include("models/blocks.jl")
include("models/attention.jl")
include("models/batched_mul_4d.jl")
include("models/UNetFixed.jl")
include("models/UNetConditioned.jl")

args = @env begin
    # graph
    noise_scale = 0.5  # n_root_nodes
    hidden = 10  # n_root_nodes
    min_depth = 5  # minimum depth of ancestors for the target node
    n_nodes = 10
    n_root_nodes = 1  # n_root_nodes
    n_anomaly_nodes = 2
    n_samples = 500  # n observations
    activation=Flux.relu
    batchsize = 100
    d_hid = 16
    decay = 1e-5  # weight decay parameter for AdamW
    epochs = 2
    fourier_scale=30.0f0
    has_node_outliers = true  # node outlier setting
    hidden_dim = 300  # hiddensize factor
    hidden_dims = [300, 300]
    save_path = ""
    load_path = "data/main-2d.bson"
    lr = 1e-3  # learning rate
    anomaly_fraction = 0.1
    n_anomaly_samples = 100  # n faulty observations
    n_batch = 1000
    n_layers = 3
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    n_timesteps = 100
    output_dim = 2
    perturbed_scale = 1f0
    seed = 1  #  random seed
    to_device = Flux.gpu
    nσ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
    ε = 1f-5
end

function sample_natural_number(; init_mass)
    current_mass = init_mass
    probability = rand()
    k = 1
    is_searching = true
    while is_searching
        if probability <= current_mass
            return k
        else
            k += 1
            current_mass += 1 / (k ^ 2)
        end
    end
end

""" target_node is the last node
n_root_nodes = 1
n_downstream_nodes = 1
activation = Flux.relu
"""
function random_mlp_dag_generator(; n_root_nodes, n_downstream_nodes, noise_scale, hidden, dist, activation)
    @info "random_nonlinear_dag_generator"
    dag = BayesNet()
    for i in 1:n_root_nodes
        cpd = RootCPD(Symbol("X$i"), dist(0, 1))
        push!(dag, cpd)
    end
    for i in 1:n_downstream_nodes
        parents = sample(names(dag), min(sample_natural_number(init_mass=0.6), length(dag)), replace=false)
        # Random mechanism
        pa_size = length(parents)
        W1 = rand(Uniform(0.5, 2.0), (hidden, pa_size))
        W1[rand(size(W1)...) .< 0.5] .*= -1
        W2 = rand(Uniform(0.5, 2.0), (1, hidden))
        W2[rand(size(W2)...) .< 0.5] .*= -1
        mlp = Chain(
                    Dense(pa_size => hidden, activation, bias=false),
                    Dense(hidden => 1, bias=false),
                   ) |> f64
        mlp[1].weight .= W1
        mlp[2].weight .= W2
        cpd = MlpCPD(Symbol("X$(i + n_root_nodes)"), parents, mlp, dist(0.0, Float64(noise_scale)))
        push!(dag, cpd)
    end
    return dag
end

function Base.getindex(bn::BayesNet, node::Symbol)
    bn.cpds[bn.name_to_index[node]]
end

"""
TODO? return data, noise, and ∇noise
"""
function draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args)
    d = length(g.cpds)
    #-- normal data
    ε = sample_noise(g, args.n_samples)
    x = forward(g, ε)

    #-- perturbed data, 3σ
    g′ = deepcopy(g)
    for cpd = g′.cpds
        if isempty(parents(cpd))  # perturb root nodes
            cpd.d = typeof(cpd.d)(cpd.d.μ, args.perturbed_scale * scale(cpd.d))
        end
    end
    ε′ = sample_noise(g′, args.n_samples)
    x′ = forward(g′, ε′)

    #-- select anomaly nodes
    ga = deepcopy(g)
    anomaly_nodes = sample(1:d, n_anomaly_nodes)
    a = anomaly_nodes |> first
    for a in anomaly_nodes
        ga.cpds[a].d = Uniform(3, 5)
    end
    #-- anomaly data
    εa = sample_noise(ga, args.n_anomaly_samples)
    xa = forward(ga, εa)

    xμ = x - ε
    xμ′ = x′ - ε′
    xμa = xa - εa
    return ε, x, ε′, x′, εa, xa, xμ, xμ′, xμa
end

"""
scale, hidden = 0.5, 100
n_nodes=round(Int, min_depth^2 / 3 + 1)
n_root_nodes=rand(1, n_nodes÷5+1)
scale=0.5
hidden=100
"""
function get_data(; args)
    @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_scale, hidden, activation = args
    n_downstream_nodes = n_nodes - n_root_nodes
    g = random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, noise_scale, hidden, activation)
    g, draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args);
end

function plot1(j; g, x, xμ, ε, μx, σx)
    @show j
    cpd = g.cpds[j]
    paj = @> g.dag adjacency_matrix Matrix{Bool} getindex(:, j)
    x_paj = x[paj, :]
    x_j = x[[j], :]
    ε_j = ε[[j], :]
    @> ε_j maximum, minimum, mean, std
    xμ_j = xμ[[j], :]
    μ_j = μx[[j], :]
    σ_j = σx[[j], :]
    μ_paj = μx[paj, :]
    σ_paj = σx[paj, :]
    x_dim = size(x_paj, 1)
    @show j x_dim
    if x_dim == 1
        plot2d(cpd, x_paj, μ_paj, σ_paj, xμ_j, x_j, μ_j, σ_j, j)
    elseif x_dim == 2
        plot3d(cpd, x_paj, μ_paj, σ_paj, xμ_j, x_j, μ_j, σ_j, j)
    else
        plot(title="more than 2d")
    end
end

function plot2d(cpd, x_paj, μ_paj, σ_paj, xμ_j, x_j, μ_j, σ_j, j)
    ys = forward_scaled(cpd, x_paj, μ_paj, σ_paj, μ_j, σ_j)
    y = forward(cpd, x_paj)
    #-- defaults
    xlim = ylim = zlim = (-3, 3)
    # default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, zlim, color=:seaborn_deep, markersize=2, leg=nothing)
    default(; aspect_ratio=:equal, grid=true, xlim, ylim, zlim, markersize=2, markerstrokewidth=0)
    # , leg=nothing, color=:seaborn_deep,
    #-- plot data
    x1, = eachrow(x_paj);
    x3 = zero(x1);
    @≥ x_j, ys, y vec.()
    pl_data = scatter(x1, x3; lab="input", c=colors[1], xlab=L"dim_1", ylab=L"dim_2", zlab=L"output", title="FCM $j")
    scatter!(pl_data, x1, vec(x_j); lab="output", c=colors[2])
    scatter!(pl_data, x1, vec(xμ_j); lab="output mean", c=colors[3])
    scatter!(pl_data, x1, vec(ys); lab="forward_scaled", c=colors[4])
    scatter!(pl_data, x1, vec(y); lab="forward", c=colors[5])
    pl_data
end

function plot3d(cpd, x_paj, μ_paj, σ_paj, xμ_j, x_j, μ_j, σ_j, j)
    ys = forward_scaled(cpd, x_paj, μ_paj, σ_paj, μ_j, σ_j)
    y = forward(cpd, x_paj)
    #-- defaults
    xlim = ylim = zlim = (-3, 3)
    # default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, zlim, color=:seaborn_deep, markersize=2, leg=nothing)
    default(; aspect_ratio=:equal, grid=true, xlim, ylim, zlim, markersize=2, markerstrokewidth=0)
    # , leg=nothing, color=:seaborn_deep,
    #-- plot data
    x1, x2 = eachrow(x_paj);
    x3 = zero(x1);
    pl_data = scatter(x1, x2, x3; lab="input", c=colors[1], xlab=L"pc_1", ylab=L"pc_2", zlab=L"output", title="FCM $j")
    scatter!(pl_data, x1, x2, vec(x_j); lab="output", c=colors[2])
    scatter!(pl_data, x1, x2, vec(xμ_j); lab="output mean", c=colors[3])
    scatter!(pl_data, x1, x2, vec(ys); lab="forward_scaled", c=colors[4])
    scatter!(pl_data, x1, x2, vec(y); lab="forward", c=colors[5])
    pl_data
end

""" Plot independent noise dists, outliers, and 2d-PCA-FCM
"""
function plot_data(g, ε, x, ε′, x′, εa, xa, μx, σx, xμ, xμ′)
    d = @> g.dag adjacency_matrix size(1)
    @> Plots.plot(plot1.(2:d; g, x, xμ, ε, μx, σx)..., size=(1000, 800)) savefig("fig/fcms.png")
end

""" Plot independent noise dists, outliers, and 2d-PCA-FCM
"""
function plot_data_outliers(g, ε, x, ε′, x′, εa, xa, μx, σx, xμ, xμ′)
    d = @> g.dag adjacency_matrix size(1)
    @> Plots.plot(plot1.(2:d; g, x, xμ, ε, μx, σx)..., size=(1000, 800)) savefig("fig/fcms.png")
end

""" Generate and save data
5 normals, 5 laplaces
"""
function generate_data(; args)
    @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_scale, hidden, activation = args
    n_downstream_nodes = n_nodes - n_root_nodes
    dist = Normal
    # for dist in [Normal, Laplace]
    #     for _ = 1:5
            g = random_mlp_dag_generator(; n_root_nodes, n_downstream_nodes, noise_scale, hidden, dist, activation)
            ε, x, ε′, x′, εa, xa, xμ, xμ′, xμa = draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args);

            #-- normalise data
            # X = @> hcat(x, x′);
            X = x;
            μx, σx = @> X mean(dims=2), std(dims=2);
            normalise_x(x) = @. (x - μx) / σx
            scale_ε(ε) = @. ε / σx
            @≥ X, x, x′, xa, xμ, xμ′, xμa normalise_x.();
            @≥ ε, ε′, εa scale_ε.();
            @assert x ≈ xμ + ε
            @assert x′ ≈ xμ′ + ε′
            @assert xa ≈ xμa + εa

            plot_data(g, ε, x, ε′, x′, εa, xa, μx, σx, xμ, xμ′)
            gui()

#     @≥ x_j, ε_j, xμ_j, ys, y vec.()
#     df = DataFrame((; x_j, ε_j, xμ_j, ys, y))
#     sort!(df, :x_j)

#     pl_data = scatter(x1, x3; lab="input", xlab=L"pc_1", ylab=L"pc_2", zlab=L"output", title="FCM $j")
#     scatter!(pl_data, x1, vec(x_j); lab="output")
#     scatter!(pl_data, x1, vec(xμ_j); lab="output mean")
#     scatter!(pl_data, x1, vec(ys); lab="forward_scaled")
#     scatter!(pl_data, x1, vec(y); lab="forward")
#     pl_data
    # @> Plots.plot(pl_data, size=(800, 600)) savefig("fig/fcm2d.png")

        # end
    # end
end

""" Load data from saved
"""
function load_data(; args)
    @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_scale, hidden = args
    n_downstream_nodes = n_nodes - n_root_nodes
    g = random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, noise_scale, hidden)
    g, draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args);
end

generate_data(; args)
