using Revise
import Base.show, Base.eltype
import Flux._big_show, Flux._show_children
import NNlib: batched_mul
using BSON, JSON
using DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2
using Dates
using Flux
using Flux.Data: DataLoader
using Flux: crossentropy
using Optimisers
using Optimisers: Optimisers, trainable
using Plots
using Printf
using ProgressMeter
using Random

include("./imports.jl")
include("./lib/diffusion.jl")
include("./lib/graph.jl")
include("./lib/nn.jl")
include("./lib/nnlib.jl")
include("./lib/utils.jl")
include("./mlp-unet-2d.jl")
include("./random-graph-datasets.jl")
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
    activation=swish
    batchsize = 100
    d_hid = 16
    decay = 1e-5  # weight decay parameter for AdamW
    epochs = 10
    fourier_scale=30.0f0
    has_node_outliers = true  # node outlier setting
    hidden_dim = 300  # hiddensize factor
    hidden_dims = [300, 300]
    input_dim = 2
    save_path = ""
    load_path = "data/main-2d.bson"
    lr = 1e-3  # learning rate
    min_depth = 2  # minimum depth of ancestors for the target node
    anomaly_fraction = 0.1
    n_anomaly_samples = 100  # n faulty observations
    n_batch = 10_000
    n_layers = 3
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    n_root_nodes = 1  # n_root_nodes
    n_samples = 100000  # n observations
    n_timesteps = 100
    noise_scale = 1.0
    output_dim = 2
    perturbed_scale = 1f0
    seed = 1  #  random seed
    to_device = Flux.gpu
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
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

"""
n_root_nodes = 1
n_downstream_nodes = 1
activation = Flux.relu
"""
function random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, noise_scale, hidden=100, activation = Flux.relu)
    @info "random_nonlinear_dag_generator"
    dag = BayesNet()
    for i in 1:n_root_nodes
        cpd = RootCPD(Symbol("X$i"), Normal(0, 1))
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
        cpd = MlpCPD(Symbol("X$(i + n_root_nodes)"), parents, mlp, Normal(0.0, Float64(noise_scale)))
        push!(dag, cpd)
    end
    return dag
end

function Base.getindex(bn::BayesNet, node::Symbol)
    bn.cpds[bn.name_to_index[node]]
end

"""
Also return data, noise, and ∇noise
"""
function draw_normal_perturbed_anomaly(dag; args)
    #-- normal data
    normal_df = rand(dag, args.n_samples)
    # sort(normal_df, :X1)

    #-- perturbed data, 3σ
    g = deepcopy(dag)
    for cpd = g.cpds
        if isempty(parents(cpd))  # perturb root nodes
            cpd.d = Normal(cpd.d.μ, args.perturbed_scale * cpd.d.σ)
        end
    end
    perturbed_df = rand(g, args.n_samples)

    #-- select anomaly nodes
    g = deepcopy(dag)
    n_anomaly_nodes = ceil(Int, 0.1args.anomaly_fraction)
    anomaly_nodes = sample(names(g), args.n_anomaly_nodes)
    for a = anomaly_nodes
        g[a].d = Uniform(3, 5)
    end
    #-- anomaly data
    anomaly_df = rand(g, args.n_anomaly_samples)

    return normal_df, perturbed_df, anomaly_df  # drawn_noise_samples
end

include("lib/plot-utils.jl")

function plot_3data(train_df, perturbed_df, anomaly_df; xlim, ylim, fig_path="fig/3data-2d.png")
    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

    #-- plot data
    x, y = eachcol(train_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"data $(x, y)$")
    x, y = eachcol(perturbed_df)
    pl_perturbed = scatter(x, y; xlab=L"x", ylab=L"y", title=L"perturbed data $(x, y)$")
    x, y = eachcol(anomaly_df)
    pl_anomaly = scatter(x, y; xlab=L"x", ylab=L"y", title=L"anomaly data $(x, y)$")
    @> Plots.plot(pl_data, pl_perturbed, pl_anomaly; xlim, ylim, size=(1000, 800)) savefig(fig_path)
end

