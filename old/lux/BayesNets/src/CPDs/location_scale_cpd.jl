
"""
A LocationScaleCPD, returns μ(x) + σ * noise
	Assumes that target and all parents can be converted to Float64 (ie, are numeric)
	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct LocationScaleCPD{T,S} <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    μ::T  # location network
    σ::S  # scalar for scale
    pnoise::Distribution{Univariate, Continuous}  # the standard noise distribution
end

name(cpd::LocationScaleCPD) = cpd.target
parents(cpd::LocationScaleCPD) = cpd.parents
nparams(cpd::LocationScaleCPD) = 3

(cpd::LocationScaleCPD)() = (cpd)(Assignment()) # cpd()
(cpd::LocationScaleCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function (cpd::LocationScaleCPD)(a::Assignment)
    x = getindex.([a], cpd.parents)
    noise = rand(cpd.pnoise)
    y = cpd.mlp(x) + cpd.σ * noise
end

