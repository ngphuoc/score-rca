using Revise,
 CSV,
 CUDA,
 DataFrames,
 Distributions,
 EndpointRanges,
 EvalMetrics,
 Functors,
 Glob,
 JLD2,
 JSON,
 LinearAlgebra,
 Lux,
 NNlib,
 Optimisers,
 Random,
 Statistics,
 Zygote
using Parameters: @unpack

CUDA.allowscalar(false)

include("lib/utils.jl")
include("lib/graph.jl")
# include("lib/nnlib.jl")
# include("lib/prob.jl")
# include("lib/nn.jl")
# include("lib/maths.jl")

