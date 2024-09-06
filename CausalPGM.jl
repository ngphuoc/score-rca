using Optimisers
using Functors: @functor
using Parameters: @unpack
include("lib/utils.jl")

struct CausalPGM{D,P,F}
    dag::D  # upper triangular
    ps::P  # noise distributions
    fs::F  # mapping to the means, batched
end

@functor CausalPGM
@showfields CausalPGM
Optimisers.trainable(model::CausalPGM) = (; model.fs)

# function CausalPGM(; args)
#     CausalPGM(dag, ps, fs)
# end

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

""" Forward the noises throught the model following the dag topo-order
TODO: more efficient implementation
"""
function forward(model::CausalPGM, εs::AbstractMatrix{T}) where T
    @assert is_weight_masked(model)
    @unpack dag, ps, fs = model
    xs = deepcopy(εs)
    foldl(1:size(xs, 1); init=xs) do xs, i
        location = model(xs)
        ys = location + εs
        xs[i, :] .= ys[i, :]
        return xs
    end
end

""" linear coef. of forward model at ε """
function ∇ε(model::CausalPGM, εs::AbstractVecOrMat{<:Real})
    xs = forward(model, εs)

    ∇x(model, xs)
end

