using Revise
using DataFrames, CSV, Tables, BSON, JLD2, JSON, FileIO, Dates, Flux, CUDA, MLUtils, PyPlot
using Distributions, BayesNets, MultivariateStats, Flux, Optimisers, NNlib
using Flux: DataLoader, crossentropy
import Flux: _big_show, _show_children
using LaTeXStrings
using Measures
using Random, Printf, ProgressMeter, Distances
using Glob

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
    dag_generator_hidden = 10
    anomaly_fraction = 0.1
    n_anomaly_samples = 100  # n faulty observations
    activation=Flux.tanh
    has_node_outliers = true  # node outlier setting
    #-- dsm
    hidden_dim = 10  # hiddensize factor
    fourier_scale=10.0f0
    reverse_steps = 100
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
    ϵ = 1f-3
    save_path = "data/main-2d.bson"
    load_path = ""
    #-- training
    training = true
    batchsize = 50
    lr = 1e-3  # learning rate
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    n_timesteps = 10
    perturbed_scale = 1f0
    decay = 1e-5  # weight decay parameter for AdamW
    epochs = 300
    seed = 1  #  random seed
    to_device = Flux.gpu ∘ f32
end

Random.seed!(args.seed)
CUDA.seed!(args.seed)

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
activation = Flux.tanh
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

""" target_node is the last node
n_root_nodes = 1
n_downstream_nodes = 1
activation = Flux.tanh
"""
function random_mlpls_dag_generator(; min_depth, n_nodes, n_root_nodes, hidden, noise_dists, activation)
    @info "random_mlpls_dag_generator"
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
            mlpl = Chain(
                        Dense(pa_size => hidden, activation, bias=false),
                        Dense(hidden => 1, bias=false),
                       ) |> f64
            mlpl[1].weight .= W1
            mlpl[2].weight .= W2
            mlps = Chain(
                        Dense(pa_size => hidden, activation, bias=false),
                        Dense(hidden => 1, Flux.σ, bias=false),
                        x -> x .+ 0.5f0,
                       ) |> f64
            cpd = MlpLsCPD(node_names[j], parents, mlpl, mlps, noise_dists)
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
function draw_normal_perturbed_anomaly(g; args)
    n_anomaly_nodes = args.n_anomaly_nodes
    n_anomaly_samples = args.n_anomaly_samples
    d = length(g.cpds)
    #-- normal data
    z = sample_noise(g, args.n_samples)
    x, s = forward(g, z)

    #-- perturbed data, 3σ
    g3 = deepcopy(g)
    for cpd = g3.cpds
        if isempty(parents(cpd))  # perturb root nodes
            cpd.d = scale3(cpd.d)
        end
    end
    z3 = sample_noise(g3, args.n_samples)
    x3, s3 = forward(g3, z3)

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
    za = sample_noise(ga, 10000n_anomaly_samples)
    xa, sa = forward(ga, za)
    za
    ds = [cpd.d for cpd in g.cpds]
    za = @> zval.(ds, eachrow(za)) hcats transpose
    anomaly_nodes
    z2 = za[anomaly_nodes, :]
    @> abs.(z2) .> 4 minimum(dims=1) vec
    ia = @> abs.(z2) .> 4 minimum(dims=1) vec findall
    ia = ia[1:n_anomaly_samples]
    za = za[:, ia]
    sa = sa[:, ia]
    xa = xa[:, ia]

    l = x - s .* z
    l3 = x3 - s3 .* z3
    # @show @> xa, sa, za size.()
    la = xa - sa .* za
    return z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes
end

function dist_name(d::Distribution)
    s = typeof(d).name.name
    σ = scale(d)
    "$s$σ"
end

data_path(d::MixedDist; dir="data") = data_path(d.ds; dir)

function data_path(ds::AbstractVector{<:Distribution}; dir="data")
    s = @> dist_name.(ds) join
    "$dir/data-$s.bson"
end

function fig_name(args)
    "$(string(args.noise_dist))-$(args.data_id)"
end

function various_plots(g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, data_id; dir="datals")
    plots = [("Noise", z), ("Scale", s), ("Mean", l), ("Observation", x), ("Noise", za), ("Scale", sa), ("Mean", la), ("Observation", xa)]
    d = size(x, 1)
    n_plots = length(plots)
    fig, axs = subplots(n_plots, d, figsize=(6*d, 6*n_plots))
    for i = 1:d
        for k = 1:n_plots
            plot_name, data = plots[k]
            ax = axs[k, i]
            i in anomaly_nodes && k > 3 && (plot_name *= " Outlier")
            ax.set_title("$plot_name $i")
            ax.hist(data[i, :], bins=50, alpha=0.7)
        end
    end
    tight_layout()
    fname = "$dir/generated_data_skewed-$(data_id).png"
    savefig(fname)
end

function generate_datals_skewed(args)
    @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_dist, dag_generator_hidden, activation = args
    for data_id = 1:1
        # ds = map((d, s) -> d(0, s), rand([Normal, Laplace], 2), rand(0.1:0.1:1, 2))
        ds = map((d, s) -> d(0, s), rand([Normal], 2), rand(0.1:0.1:1, 2))
        noise_dists = MixedDist(ds)
        fname = data_path(noise_dists; dir="datals")
        @info "generating " * fname
        g = random_mlpls_dag_generator(; min_depth, n_nodes, n_root_nodes, hidden=dag_generator_hidden, noise_dists, activation)
        z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes = draw_normal_perturbed_anomaly(g; args);
        @show anomaly_nodes
        BSON.@save fname args g z x l s z3 x3 l3 s3 za xa la sa anomaly_nodes ds
        various_plots(g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, data_id; dir="datals")
    end
end

# function generate_data_timeit(args)
#     @unpack min_depth, n_nodes, n_root_nodes, n_anomaly_nodes, noise_dist, hidden, activation = args
#     rng = range(log(50), log(1000), length=30)
#     rng = round.(Int, exp.(rng))
#     for n_nodes in rng
#         for id = 1:1
#             ds = map((d, s) -> d(0, s), rand([Normal, Laplace], 2), rand(0.1:0.1:1, 2))
#             noise_dists = MixedDist(ds)
#             @info "generating " * data_path(noise_dists)
#             g = random_mlp_dag_generator(; min_depth, n_nodes, n_root_nodes, hidden, noise_dists, activation)
#             ε, x, y, ε3, x3, y3, εa, xa, ya, anomaly_nodes = draw_normal_perturbed_anomaly(g; args);
#             @show anomaly_nodes
#             filepath = "data/data-timeit&nodes=$n_nodes&id=$id.bson"
#             BSON.@save filepath args g ε x y ε3 x3 y3 εa xa ya anomaly_nodes ds
#         end
#     end
# end

""" Load data from saved, "/data/noise_dist-data_id.bson"
y: output mean: x ≈ y + z
return g, normal, perturb, and outlier data
dir="datals"
"""
function load_data(args; dir="datals")
    fpaths = glob("$dir/data-*.bson")
    @assert length(fpaths) > 0
    fpath = fpaths[args.data_id]
    @info "Loading " * fpath
    BSON.@load fpath g z x l s z3 x3 l3 s3 za xa la sa anomaly_nodes ds  # don't load args
    @assert x ≈ l + s .* z
    @assert x3 ≈ l3 + s3 .* z3
    @assert xa ≈ la + sa .* za
    return g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, fpath
end

# generate_data(args)
# generate_data_skewed(args)
# generate_datals_skewed(args)
# generate_data_timeit(args)
# plot_data(args)

round3(x) = round.(x, digits=3)

