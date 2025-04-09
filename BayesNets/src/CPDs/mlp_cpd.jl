"""
A linear Gaussian CPD, always returns a Normal
	Assumes that target and all parents can be converted to Float64 (ie, are numeric)
	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct MlpCPD{T} <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    mlp::T
    d::Distribution{Univariate, Continuous}
end

name(cpd::MlpCPD) = cpd.target
parents(cpd::MlpCPD) = cpd.parents
nparams(cpd::MlpCPD) = length(Flux.destructure(cpd)[1])

(cpd::MlpCPD)() = (cpd)(Assignment()) # cpd()
(cpd::MlpCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function (cpd::MlpCPD)(a::Assignment)
    x = getindex.([a], cpd.parents)
    μ = cpd.mlp(x)
    only(μ) + cpd.d
end

