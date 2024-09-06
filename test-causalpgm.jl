using Tables: matrix
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

include("lib/utils.jl")
include("lib/diffusion.jl")
include("lib/graph.jl")
include("lib/nn.jl")
include("lib/nnlib.jl")
include("CausalPGM.jl")

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
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
end

dag = Bool[0 1; 0 0]
ps = [Normal(0f0, 1f0), Normal(0f0, 3f0), ]
X, H, F = 2, 100, 2
fs = Chain(
          GroupDense(X, H, F, activation),  # all inputs
          GroupDense(H, 1, F),  # single output for each node
         )
W1 = rand(Uniform(0.5, 2.0), (H, 1));
W1[rand(size(W1)...) .< 0.5] .*= -1;
W2 = rand(Uniform(0.5, 2.0), (1, H));
W2[rand(size(W2)...) .< 0.5] .*= -1;
fs[1].weight .= 0f0;
fs[2].weight .= 0f0;
fs[1].weight[:, 1, 2] .= W1;  # fcm2 get X1 input
fs[2].weight[:, :, 2] .= W2;  # fcm2 single output for mean
model = CausalPGM(dag, ps, fs)

εs = sample_noise(model, 10)
forward(model, εs)

