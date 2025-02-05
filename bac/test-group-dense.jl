include("imports.jl")
using Revise
using DataFrames, Distributions, BayesNets, CSV, Tables
using BayesNets: plot, name
using DataFrames: index
using Graphs, GraphPlot
using Revise
using DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2
using Optimisers, BSON
using ProgressMeter: Progress, next!
using CUDA
using Flux
using Flux: gpu, Chain, Dense, relu, DataLoader
include("bayesnets-extra.jl")
include("group-mlp-unet.jl")
include("group-mlp-regression.jl")
include("lib/diffusion.jl")

using CUDA, BenchmarkTools


X, Y, F, batchsize = 10, 20, 30, 128;
m = GroupDense(X, Y, F) |> gpu;
x = randn(X, batchsize) |> gpu;

m(x);

# 95 μs
@btime o = m(x);
# 480 μs
@btime gradient(m -> sum(m(x)), m, );

c = Chain(
          Reshape(1, X*F, :),
          Conv((1,), X*F=>Y*F, groups=F),
          Reshape(Y, F, :),
         ) |> gpu;

x3 = @> unsqueeze(x, 2) repeat(outer=(1, F, 1))
c(x3);

# 80 μs
@btime begin
    x3 = @> unsqueeze(x, 2) repeat(outer=(1, F, 1))
    o = c(x3);
end;

# 545 μs
@btime begin
    x3 = @> unsqueeze(x, 2) repeat(outer=(1, F, 1))
    gradient(m -> sum(m(x3)), m, );
end;

