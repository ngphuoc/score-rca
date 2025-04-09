"""
A linear Gaussian CPD, always returns a Normal
	Assumes that target and all parents can be converted to Float64 (ie, are numeric)
	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct MlpLsCPD{T, U} <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    mlpl::T
    mlps::U
    d::Distribution{Univariate, Continuous}
end

name(cpd::MlpLsCPD) = cpd.target
parents(cpd::MlpLsCPD) = cpd.parents
nparams(cpd::MlpLsCPD) = length(Flux.destructure(cpd)[1])

(cpd::MlpLsCPD)() = (cpd)(Assignment()) # cpd()
(cpd::MlpLsCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function (cpd::MlpLsCPD)(a::Assignment)
    x = getindex.([a], cpd.parents)
    l = cpd.mlpl(x)
    s = cpd.mlps(x)
    only(l) + s * cpd.d
end

