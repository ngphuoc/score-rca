import Base.show, Base.eltype
import Flux._big_show, Flux._show_children
import NNlib: batched_mul
using Parameters: @unpack
using BSON, JSON, DataFrames, Distributions, BayesNets, CSV, FileIO, JLD2, Dates, Flux, Optimisers
using Flux: DataLoader, crossentropy
using Optimisers: Optimisers, trainable
using Plots, Printf, ProgressMeter, Random
include("lib/utils.jl")
include("lib/graph.jl")
include("lib/outlier.jl")

include("./lib/diffusion.jl")
include("./lib/graph.jl")
# include("./lib/nn.jl")
include("./lib/nnlib.jl")
include("./lib/utils.jl")
# include("./mlp-unet-2d.jl")
# include("./random-graph-datasets.jl")
# include("./rca-data.jl")
include("./rca.jl")
# include("datasets.jl")
include("utilities.jl")
include("models/embed.jl")
include("models/ConditionalChain.jl")
include("models/blocks.jl")
include("models/attention.jl")
include("models/batched_mul_4d.jl")
include("models/UNetFixed.jl")
include("models/UNetConditioned.jl")

args = @env begin
    activation=Flux.swish
    batchsize = 100
    d_hid = 16
    decay = 1e-5  # weight decay parameter for AdamW
    epochs = 1
    fourier_scale=30.0f0
    has_node_outliers = true  # node outlier setting
    hidden_dim = 300  # hiddensize factor
    hidden_dims = [300, 300]
    input_dim = 2
    save_path = "data/main-2d.bson"
    load_path = ""
    lr_regressor = 1e-3  # learning rate
    lr_unet = 1e-3  # learning rate
    min_depth = 2  # minimum depth of ancestors for the target node
    n_anomaly_nodes = 2
    n_anomaly_samples = 100  # n faulty observations
    n_batch = 10_000
    n_layers = 3
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    n_root_nodes = 1  # n_root_nodes
    n_samples = 1000  # n observations
    n_timesteps = 100
    noise_scale = 1.0
    output_dim = 2
    perturbed_scale = 1f0
    seed = 1  #  random seed
    to_device = Flux.gpu
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
end

function spirals_2d()
    X = normalize_neg_one_to_one(make_spiral(n_batch))
    X_val = normalize_neg_one_to_one(make_spiral(floor(Int, 0.1 * n_batch)))
    loader = Flux.DataLoader((X,) |> gpu; batchsize=32, shuffle=true);
    val_loader = Flux.DataLoader((X_val,) |> gpu; batchsize=32, shuffle=false);
    (x,) = @> loader first gpu
    d = size(x, 1)
end

function spirals_python()
    datetime_prefix = @> string(now())[1:16] replace(":"=>"h")
    X = @> sample_2d("2spirals", args.n_samples) PyArray transpose Array;
    @> X minimum, maximum, mean, std
    return X
end

