"""
A nonlinear Gaussian CPD, always returns a Normal

	Assumes that target and all parents can be converted to Float64 (ie, are numeric)

    P(x|parents(x)) = Normal(μ=mlp(parents(x)) , σ)
"""
mutable struct NonlinearScoreCPD{F,G} <: CPD{Normal} where {F,G}
    target::NodeName
    parents::NodeNames
    mlp::F
    score::G
    σ::Float64
end
NonlinearScoreCPD(target::NodeName, μ::Float64, σ::Float64) = NonlinearScoreCPD(target, NodeName[], Float64[], μ, σ)

name(cpd::NonlinearScoreCPD) = cpd.target
parents(cpd::NonlinearScoreCPD) = cpd.parents
nparams(cpd::NonlinearScoreCPD) = length(cpd.mlp) + 2

function (cpd::NonlinearScoreCPD)(x::Assignment)
    x = @> getindex.([x], cpd.parents) Vector{Float32}
    μ = cpd.mlp(x) |> only
    Normal(μ, cpd.σ)
end

(cpd::NonlinearScoreCPD)() = (cpd)(Assignment()) # cpd()
(cpd::NonlinearScoreCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

