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
    #-- graph
    data_id = 1  # Normal	Laplace	Student-t	Gumbel	Fréchet	Weibull
    noise_dist = Normal  # Normal	Laplace	Student-t	Gumbel	Fréchet	Weibull
    noise_scale = 1.0  # n_root_nodes
    hidden = 10  # n_root_nodes
    min_depth = 3  # minimum depth of ancestors for the target node
    n_nodes = 4
    n_root_nodes = 1  # n_root_nodes
    n_anomaly_nodes = 2
    n_samples = 200  # n observations
    anomaly_fraction = 0.1
    n_anomaly_samples = 20  # n faulty observations
    activation=Flux.relu
    has_node_outliers = true  # node outlier setting
    #-- dsm
    fourier_scale=10.0f0
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
    hidden_dim = 10  # hiddensize factor
    save_path = "data/main-2d.bson"
    load_path = ""
    #-- training
    batchsize = 50
    lr = 1e-3  # learning rate
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    n_timesteps = 100
    perturbed_scale = 1f0
    decay = 1e-5  # weight decay parameter for AdamW
    epochs = 2
    seed = 1  #  random seed
    to_device = Flux.cpu ∘ f32
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
function random_mlp_dag_generator(; n_root_nodes, n_downstream_nodes, noise_scale, hidden, noise_dist, activation)
    @info "random_nonlinear_dag_generator"
    dag = BayesNet()
    for i in 1:n_root_nodes
        cpd = RootCPD(Symbol("X$i"), noise_dist(0, 1))
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
        cpd = MlpCPD(Symbol("X$(i + n_root_nodes)"), parents, mlp, noise_dist(0.0, Float64(noise_scale)))
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
    anomaly_nodes = sample(1:d, n_anomaly_nodes, replace=false)
    a = anomaly_nodes |> first
    for a in anomaly_nodes
        ga.cpds[a].d = Uniform(3, 5)
    end
    #-- anomaly data
    εa = sample_noise(ga, args.n_anomaly_samples)
    xa = forward(ga, εa)

    y = x - ε
    y′ = x′ - ε′
    ya = xa - εa
    return ε, x, y, ε′, x′, y′, εa, xa, ya, anomaly_nodes
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

function plot1(j; g, ε, x, y, μx, σx, εa, xa, ya, anomaly_nodes)
    @show j
    cpd = g.cpds[j]
    paj = @> g.dag adjacency_matrix Matrix{Bool} getindex(:, j)
    parent_child(x) = x[paj, :], x[[j], :]
    εs, xs, ys, μs, σs = @> ε, x, y, μx, σx parent_child.()
    εas, xas, yas = @> εa, xa, ya parent_child.()
    plot23d_pca(cpd, εs, xs, ys, μs, σs, εas, xas, yas, j, anomaly_nodes)
end

""" input can be 1 or 2d
"""
function plot23d_pca(cpd, εs, xs, ys, μs, σs, εas, xas, yas, j, anomaly_nodes)
    y = forward_scaled(cpd, xs, μs, σs)
    ya = forward_scaled(cpd, xas, μs, σs)
    #-- defaults
    xlim = ylim = zlim = (-5, 5)
    default(; aspect_ratio=:equal, grid=true, xlim, ylim, zlim, markersize=2, markerstrokewidth=0, titlefontsize=12, fontfamily="Computer Modern", framestyle=:box)
    x12 = xs[1]
    xa12 = xas[1]
    s = j in anomaly_nodes ? "abnormal" : ""
    title = L"f_{%$j} %$s"
    if size(x12, 1) > 2
        @info "more than 2d"
        title = "PCA $j $s"
        pca = fit(PCA, x12; maxoutdim=2)
        x12 = predict(pca, x12)
        xa12 = predict(pca, xa12)
    end
    x12 = eachrow(x12);
    z3 = zero(xs[2]);  # 1d, for input points at base
    xa12 = eachrow(xa12);
    za3 = zero(xas[2])
    x3 = xs[2]
    xa3 = xas[2]
    @≥ x3, z3, xa3, za3, y, ya vec.()
    pl_data = scatter(x12..., z3; lab="input", leg=nothing, c=colors[1], xlab=L"dim_1", ylab=L"dim_2", zlab=L"output", title)
    scatter!(pl_data, xa12..., za3; lab="input outliers", leg=nothing, c=colors[2])
    scatter!(pl_data, x12..., x3; lab="output", leg=nothing, c=colors[3])
    scatter!(pl_data, xa12..., xa3; lab="output outliers", leg=nothing, c=colors[4])
    scatter!(pl_data, x12..., y; lab="output mean", leg=nothing, c=colors[5])
    scatter!(pl_data, xa12..., ya; lab="output mean outliers", leg=nothing, c=colors[6])
    pl_data
end

function data_path(args)
    "data/data-$(string(args.noise_dist))-$(args.data_id).bson"
end

function fig_name(args)
    "$(string(args.noise_dist))-$(args.data_id)"
end

""" Generate and save data
5 normals, 5 laplaces
"""
function generate_data(args)
    @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_scale, noise_dist, hidden, activation = args
    n_downstream_nodes = n_nodes - n_root_nodes
    for noise_dist in [Normal, Laplace]
        for data_id = 1:5
            g = random_mlp_dag_generator(; n_root_nodes, n_downstream_nodes, noise_scale, hidden, noise_dist, activation)
            ε, x, y, ε′, x′, y′, εa, xa, ya, anomaly_nodes = draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args);
            @show anomaly_nodes
            BSON.@save data_path(args) args g ε x y ε′ x′ y′ εa xa ya anomaly_nodes
        end
    end
end

function plot_data(args)
    @info "Loading data " * data_path(args)
    BSON.@load data_path(args) g ε x y ε′ x′ y′ εa xa ya anomaly_nodes  # don't load args
    @> x mean, std
    @> xa mean, std
    #-- normalise data
    # X = @> hcat(x, x′);
    X = x;
    μx, σx = @> X mean(dims=2), std(dims=2);
    normalise_x(x) = @. (x - μx) / σx
    scale_ε(ε) = @. ε / σx
    @≥ X, x, x′, xa, y, y′, ya normalise_x.();
    @≥ ε, ε′, εa scale_ε.();
    @assert x ≈ y + ε
    @assert x′ ≈ y′ + ε′
    @assert xa ≈ ya + εa
    # plot_data(; g, ε, x, y, μx, σx, εa, xa, aμ)
    d = @> g.dag adjacency_matrix size(1)
    x1 = x[1, :]
    xa1 = xa[1, :]
    title = ""
    s = 1 in anomaly_nodes ? "abnormal" : ""
    title = "root $s"
    p1 = scatter(x1, zero(x1); lab="input", leg=nothing, c=colors[1], xlab=L"dim_1", ylab=L"dim_2", zlab=L"output", title)
    scatter!(p1, xa1, zero(xa1); lab="input outliers", leg=nothing, c=colors[2])
    ps = plot1.(2:d; g, ε, x, y, μx, σx, εa, xa, ya, anomaly_nodes)
    labels = ["input" "input outliers" "output" "output outliers" "output mean" "output mean outliers"]
    n = length(labels)
    l = @layout [Plots.grid(2, 2) a{0.2w}]
    p0 = scatter([-1], [-1], c=colors[1], lims=(0,1), legendfontsize=7, legend=:left, label=labels[1], frame=:none);
    [scatter!(p0, [-1], [-1], c=colors[i], label=labels[i]) for i=2:n]
    # color_pallete
    @> Plots.plot(p1, ps..., p0, layout=l, size=(1200, 800)) savefig("fig/fcm-outliers-$(fig_name(args)).png")
end

""" Load data from saved, "/data/noise_dist-data_id.bson"
y: output mean: x ≈ y + ε
return g, normal, perturb, and outlier data
"""
function load_normalised_data(args)
    @info "Loading data " * data_path(args)
    BSON.@load data_path(args) g ε x y ε′ x′ y′ εa xa ya anomaly_nodes  # don't load args
    @> x mean, std
    @> xa mean, std
    #-- normalise data
    # X = @> hcat(x, x′);
    X = x;
    μx, σx = @> X mean(dims=2), std(dims=2);
    normalise_x(x) = @. (x - μx) / σx
    scale_ε(ε) = @. ε / σx
    @≥ X, x, x′, xa, y, y′, ya normalise_x.();
    @≥ ε, ε′, εa scale_ε.();
    @assert x ≈ y + ε
    @assert x′ ≈ y′ + ε′
    @assert xa ≈ ya + εa
    # plot_data(; g, ε, x, y, μx, σx, εa, xa, aμ)
    d = @> g.dag adjacency_matrix size(1)
    x1 = x[1, :]
    xa1 = xa[1, :]
    return g, x, x′, xa, y, y′, ya, ε, ε′, εa, μx, σx, anomaly_nodes
end

# generate_data(args)
# plot_data(args)

