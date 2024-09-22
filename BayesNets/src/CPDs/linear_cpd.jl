"""
A linear Gaussian CPD, always returns a Normal
	Assumes that target and all parents can be converted to Float64 (ie, are numeric)
	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct LinearCPD{T} <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    a::T
    d::Distribution{Univariate, Continuous}
end

name(cpd::LinearCPD) = cpd.target
parents(cpd::LinearCPD) = cpd.parents
nparams(cpd::LinearCPD) = 2

(cpd::LinearCPD)() = (cpd)(Assignment()) # cpd()
(cpd::LinearCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function (cpd::LinearCPD)(a::Assignment)
    x = getindex.([a], cpd.parents)
    μ = cpd.a(x)
    only(μ) + cpd.d
end

