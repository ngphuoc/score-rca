"""
A nonlinear Gaussian CPD, always returns a Normal

	Assumes that target and all parents can be converted to Float64 (ie, are numeric)

    P(x|parents(x)) = Normal(μ=mlp(parents(x)) , σ)
"""
mutable struct NonlinearGaussianCPD{T} <: CPD{Normal} where T
    target::NodeName
    parents::NodeNames
    mlp::T
    σ::Float64
end
NonlinearGaussianCPD(target::NodeName, μ::Float64, σ::Float64) = NonlinearGaussianCPD(target, NodeName[], Float64[], μ, σ)

name(cpd::NonlinearGaussianCPD) = cpd.target
parents(cpd::NonlinearGaussianCPD) = cpd.parents
nparams(cpd::NonlinearGaussianCPD) = length(cpd.mlp) + 2

function (cpd::NonlinearGaussianCPD)(x::Assignment)
    x = @> getindex.([x], cpd.parents) Vector{Float32}
    μ = cpd.mlp(x) |> only
    Normal(μ, cpd.σ)
end

(cpd::NonlinearGaussianCPD)() = (cpd)(Assignment()) # cpd()
(cpd::NonlinearGaussianCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

