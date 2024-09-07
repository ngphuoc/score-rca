using CUDA: functional
using Optimisers
using Functors: @functor
using Parameters: @unpack
include("lib/utils.jl")

struct CausalPGM{D,P,F}
    dag::D  # upper triangular
    ps::P  # noise distributions
    fs::F  # mapping to the means
end

@functor CausalPGM
@showfields CausalPGM
Optimisers.trainable(model::CausalPGM) = (; model.fs)

# function CausalPGM(; args)
#     CausalPGM(dag, ps, fs)
# end

""" For @assertion in forward mode to check if weights have been masked or not"""
function is_weight_masked(model)
    sum(model.fs[1].weight[:, 2, 2]) == 0f0  # diagonal weight (self loop) is zero
end

""" Set zero masks to batched weights instead of input: (x*mask)*weight = x*(mask*weight)
-> apply this after each parameter update
-> need an @assertion in forward mode to check if weights have been masked or not
"""
function mask_weight!(model::CausalPGM)
    @unpack dag, ps, fs = model
    xs = zero(εs)
    for i = 1:size(dag, 1)  # loop fcm i
        fs[1][:, :, i] .*= dag[:, i]  # mask input layer only
    end
end

""" Sample zero mean noises from CausalPGM.ps """
function sample_noise(model::CausalPGM, n_samples::Int)
    @> rand.(model.ps, n_samples) transpose.() vcats
end

""" One-step mapping throught the model, return the location """
function (model::CausalPGM)(x::AbstractVecOrMat{<:Real})
    @unpack dag, ps, fs = model
    xb = @> unsqueeze(x, 2) repeat(outer=(1, size(dag, 1)))
    yb = @> fs(xb) squeeze(1)
end

""" Forward the noises throught the model following the dag topo-order
TODO: more efficient implementation
"""
function forward(model::CausalPGM, εs::AbstractMatrix{T}, ys::AbstractVector{T}) where T
    # @assert is_weight_masked(model)
    @unpack dag, ps, fs = model
    d = size(dag, 1)
    for i = 1:d
        εs[i, :] .+= model(εs)[i, :]
    end
    ys[1] = sum(εs[d, :])
    return nothing
end

""" linear coef. of forward model at ε """
function ∇ε(model::CausalPGM, εs::AbstractVecOrMat{<:Real})
    xs = forward(model, εs)
    gradient(Reverse, rosenbrock_inp, [1.0, 2.0])

    ∇x(model, xs)
end

function test_enzyme()
    ε = sample_noise(model, 10)
    bε = zero(ε)
    by = fill!(similar(ε, 1), 1)  # seed 1 like back(1)
    y = zero(by)
    forward(model, ε, y)

    function ε_func_enzyme(model::CausalPGM, ε::AbstractArray{T}, y::AbstractArray{T}) where T
        @unpack dag, ps, fs = model
        d = size(dag, 1)
        for j = 1:d
            ε[[j], :] += fs[j](ε)
        end
        y[1] = sum(ε[end, :])
        nothing
    end

    ε = sample_noise(model, 10)
    bε = zero(ε)
    y = fill!(similar(ε, 1), 0)
    by = fill!(similar(y), 1)
    Enzyme.autodiff(Reverse, ε_func_enzyme, Const(model), Duplicated(ε, bε), Duplicated(y, by));

    function f(x::Array{Float64}, y::Array{Float64})
        y[1] = x[1] * x[1] + x[2] * x[1]
        return nothing
    end;

    x  = [2.0, 2.0]
    bx = [0.0, 0.0]
    y  = [0.0]
    by = [1.0];
    Enzyme.autodiff(Reverse, f, Duplicated(x, bx), Duplicated(y, by));
end

