"""
A linear Gaussian CPD, always returns a Normal
	Assumes that target and all parents can be converted to Float64 (ie, are numeric)
	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct MlpGaussianCPD{T} <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    mlp::T
    σ::Float64
end

name(cpd::MlpGaussianCPD) = cpd.target
parents(cpd::MlpGaussianCPD) = cpd.parents
nparams(cpd::MlpGaussianCPD) = 2

function (cpd::MlpGaussianCPD)(a::Assignment)
    x = getindex.([a], cpd.parents)
    μ = cpd.mlp(x)
    Normal(only(μ), cpd.σ)
end

