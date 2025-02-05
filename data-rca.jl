# Package management
using Revise

# Data manipulation and file I/O
using DataFrames, CSV, Tables, BSON, JLD2, JSON, FileIO

# Date and time utilities
using Dates

# Statistical analysis and machine learning
using Distributions, BayesNets, MultivariateStats, Flux, Optimisers, NNlib
using Flux: DataLoader, crossentropy
import Flux: _big_show, _show_children

# Visualization
using Plots
using Plots: plot, palette
using LaTeXStrings, Plots.PlotMeasures
using ColorSchemes

# Utilities
using Random, Printf, ProgressMeter, Distances
using Glob

# Custom configurations
color_palette = palette(:Paired_10)
colors = color_palette.colors

include("./lib/utils.jl")
include("./lib/graph.jl")
include("./lib/outlier.jl")
include("./lib/diffusion.jl")
include("./lib/graph.jl")
include("./lib/nnlib.jl")
include("./lib/distributions.jl")
include("./rca-graph.jl")
include("./rca.jl")
# include("utilities.jl")
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
    noise_dist = "Normal"  # Normal	Laplace	Student-t	Gumbel	Fréchet	Weibull
    min_depth = 5  # minimum depth of ancestors for the target node
    n_nodes = 15
    n_root_nodes = 1  # n_root_nodes
    n_anomaly_nodes = 2
    n_samples = 5000  # n observations
    hidden = 10
    anomaly_fraction = 0.1
    n_anomaly_samples = 500  # n faulty observations
    activation=Flux.relu
    has_node_outliers = true  # node outlier setting
    #-- dsm
    hidden_dim = 10  # hiddensize factor
    fourier_scale=10.0f0
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
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
function random_mlp_dag_generator(; min_depth, n_nodes, n_root_nodes, hidden, noise_dists, activation)
    @info "random_nonlinear_dag_generator"
    dag = random_rca_dag(min_depth, n_nodes, n_root_nodes)
    B = @> dag adjacency_matrix Matrix{Bool}
    d = nv(dag)
    dag = BayesNet()
    node_names = [Symbol("X$i") for i=1:d]
    for j = 1:d
        if sum(B[:, j]) == 0  # root
            cpd = RootCPD(node_names[j], noise_dists)
            push!(dag, cpd)
        else  # leaf
            ii = findall(B[:, j])
            parents = node_names[ii]
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
            cpd = MlpCPD(node_names[j], parents, mlp, noise_dists)
            push!(dag, cpd)
        end
    end
    return dag
end

function copy_linear_dag(bn)
    g = deepcopy(bn)
    for i = 1:length(g.cpds)
        cpd = g.cpds[i]
        if length(cpd.parents) > 0
            a = randn(length(cpd.parents))
            g.cpds[i] = LinearCPD(cpd.target, cpd.parents, a, cpd.d)
        end
    end
    g
end

function copy_bayesian_dag(bn)
    g = deepcopy(bn)
    for i = 1:length(g.cpds)
        cpd = g.cpds[i]
        if length(cpd.parents) > 0
            g.cpds[i] = LinearBayesianCPD(cpd.target, cpd.parents)
        end
    end
    g
end

function Base.getindex(bn::BayesNet, node::Symbol)
    bn.cpds[bn.name_to_index[node]]
end

function scale3(d::MixedDist, s=3)
    MixedDist(scale3.(d.ds))
end

function scale3(d, s=3)
    μ, σ = Distributions.params(d)
    typeof(d)(μ, s * σ)
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
    g3 = deepcopy(g)
    for cpd = g3.cpds
        if isempty(parents(cpd))  # perturb root nodes
            cpd.d = scale3(cpd.d)
        end
    end
    ε3 = sample_noise(g3, args.n_samples)
    x3 = forward(g3, ε3)

    #-- select anomaly nodes
    ga = deepcopy(g)
    anomaly_nodes = sample(1:d, n_anomaly_nodes, replace=false)
    a = anomaly_nodes |> first
    for a in anomaly_nodes
        # ga.cpds[a].d = Uniform(3, 5)
        cpd = ga.cpds[a]
        cpd.d = scale3(cpd.d)
    end

    #-- anomaly data
    εa = sample_noise(ga, 20args.n_anomaly_samples)
    xa = forward(ga, εa)
    εa
    ds = [cpd.d for cpd in g.cpds]
    za = @> zval.(ds, eachrow(εa)) hcats transpose
    anomaly_nodes
    z2 = za[anomaly_nodes, :]
    @> abs.(z2) .> 3 minimum(dims=1) vec
    ia = @> abs.(z2) .> 3 minimum(dims=1) vec findall
    ia = ia[1:args.n_anomaly_samples]
    εa = εa[:, ia]
    xa = xa[:, ia]
    y = x - ε
    y3 = x3 - ε3
    ya = xa - εa
    return ε, x, y, ε3, x3, y3, εa, xa, ya, anomaly_nodes
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

function dist_name(d::Distribution)
    s = typeof(d).name.name
    σ = scale(d)
    "$s$σ"
end

data_path(d::MixedDist) = data_path(d.ds)

function data_path(ds::AbstractVector{<:Distribution})
    s = @> dist_name.(ds) join
    "data/data-$s.bson"
end

# ds = [Normal(0, 1), Laplace(0, 0.5)]
# d = ds[1]
# data_path(d::T, id) where T<:Union{String, Symbol} = "data/data-$d-$id.bson"

function fig_name(args)
    "$(string(args.noise_dist))-$(args.data_id)"
end

function generate_data_skewed(args)
    @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_dist, hidden, activation = args
    for _ = 1:5
        ds = map((d, s) -> d(0, s), rand([Normal, Laplace], 2), rand(0.1:0.1:1, 2))
        noise_dists = MixedDist(ds)
        @info "generating " * data_path(noise_dists)
        g = random_mlp_dag_generator(; min_depth, n_nodes, n_root_nodes, hidden, noise_dists, activation)
        ε, x, y, ε3, x3, y3, εa, xa, ya, anomaly_nodes = draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args);
        @show anomaly_nodes
        BSON.@save data_path(noise_dists) args g ε x y ε3 x3 y3 εa xa ya anomaly_nodes ds
    end
end

function generate_data_timeit(args)
    @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_dist, hidden, activation = args
    rng = range(log(50), log(1000), length=30)
    rng = round.(Int, exp.(rng))
    for n_nodes in rng
        for id = 1:5
            ds = map((d, s) -> d(0, s), rand([Normal, Laplace], 2), rand(0.1:0.1:1, 2))
            noise_dists = MixedDist(ds)
            @info "generating " * data_path(noise_dists)
            g = random_mlp_dag_generator(; min_depth, n_nodes, n_root_nodes, hidden, noise_dists, activation)
            ε, x, y, ε3, x3, y3, εa, xa, ya, anomaly_nodes = draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args);
            @show anomaly_nodes
            filepath = "data/data-timeit&nodes=$n_nodes&id=$id.bson"
            BSON.@save filepath args g ε x y ε3 x3 y3 εa xa ya anomaly_nodes ds
        end
    end
end

function plot_data(args)
    @info "Loading " * data_path(args)
    BSON.@load data_path(args) g ε x y ε3 x3 y3 εa xa ya anomaly_nodes  # don't load args
    @> x mean, std
    @> xa mean, std
    #-- normalise data
    # X = @> hcat(x, x3);
    X = x;
    μx, σx = @> X mean(dims=2), std(dims=2);
    normalise_x(x) = @. (x - μx) / σx
    scale_ε(ε) = @. ε / σx
    @≥ X, x, x3, xa, y, y3, ya normalise_x.();
    @≥ ε, ε3, εa scale_ε.();
    @assert x ≈ y + ε
    @assert x3 ≈ y3 + ε3
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
    fpaths = glob("data/*.bson")
    @assert length(fpaths) > 0
    fpath = fpaths[args.data_id]
    @info "Loading " * fpath
    BSON.@load fpath g ε x y ε3 x3 y3 εa xa ya anomaly_nodes ds  # don't load args
    @> x mean, std
    @> xa mean, std
    #-- normalise data
    # X = @> hcat(x, x3);
    X = x;
    μx, σx = @> X mean(dims=2), std(dims=2);
    normalise_x(x) = @. (x - μx) / σx
    scale_ε(ε) = @. ε / σx
    @≥ X, x, x3, xa, y, y3, ya normalise_x.();
    @≥ ε, ε3, εa scale_ε.();
    @assert x ≈ y + ε
    @assert x3 ≈ y3 + ε3
    @assert xa ≈ ya + εa
    # plot_data(; g, ε, x, y, μx, σx, εa, xa, aμ)
    d = @> g.dag adjacency_matrix size(1)
    x1 = x[1, :]
    xa1 = xa[1, :]
    return g, x, x3, xa, y, y3, ya, ε, ε3, εa, μx, σx, anomaly_nodes
end

# generate_data(args)
generate_data_skewed(args)
# generate_data_timeit(args)
# plot_data(args)
