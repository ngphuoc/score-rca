
"""
A LocationCPD, returns μ(x) + noise
	Assumes that target and all parents can be converted to Float64 (ie, are numeric)
	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct LocationCPD{T} <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    μ::T  # location network
    d::Distribution{Univariate, Continuous}  # the standard noise distribution
end

name(cpd::LocationCPD) = cpd.target
parents(cpd::LocationCPD) = cpd.parents
nparams(cpd::LocationCPD) = 2

(cpd::LocationCPD)() = (cpd)(Assignment()) # cpd()
(cpd::LocationCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function (cpd::LocationCPD)(a::Assignment)
    x = getindex.([a], cpd.parents)
    only(cpd.μ(x)) + cpd.d
end

function forward(cpd::LocationCPD, x::AbstractArray{T}) where T
     cpd.μ(x)
end

function forward(cpd::LocationCPD, a::Assignment)
    x = getindex.([a], cpd.parents)
    forward(cpd, x)
end

