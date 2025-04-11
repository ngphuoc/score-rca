using Parameters: @unpack
include("BayesianLinearRegression.jl")

"""
μ linear Gaussian CPD, always returns μ Normal
	Assumes that target and all parents can be converted to Float64 (ie, are numeric)
	P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
"""
mutable struct LastLinearBayesianCPD{T} <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    m::BayesianLinReg
    ps::Vector{Any}  # parameters of m
    b::Float64  # for injecting outlier
    mlpl::T
end

function LastLinearBayesianCPD(target::NodeName, parents::NodeNames; α0=1.0, β0=1.0, update_prior=false, update_noise=true)
    d = length(parents)
    @assert d > 0
    m, ps = BayesianLinReg(d; α0, β0, update_prior, update_noise)
    LastLinearBayesianCPD(target, parents, m, ps, 0.0)
end

LastLinearBayesianCPD(target, parents, m, ps) = LastLinearBayesianCPD(target, parents, m, ps, 0.0)

name(cpd::LastLinearBayesianCPD) = cpd.target
parents(cpd::LastLinearBayesianCPD) = cpd.parents
nparams(cpd::LastLinearBayesianCPD) = length(cpd.μ) + length(cpd.Λ)

# Assignments of nodes is a Dict
function (cpd::LastLinearBayesianCPD)(x::Assignment)
    @unpack target, parents, m, ps, b = cpd
    μ, Λ, α, β = ps
    σ = 1 / β
    Σ = inv(Λ)
    w = rand(MvNormal(μ, Symmetric(Σ)))
    # y = getindex.([x], parents) * w
    # y = 0.0
    y = b
    for (i, p) in enumerate(cpd.parents)
        y += x[p]*w[i]
    end
    Normal(y, σ)
end
(cpd::LastLinearBayesianCPD)() = (cpd)(Assignment()) # cpd()
(cpd::LastLinearBayesianCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:μ=>1)

""" Use Distributions.fit interface
"""
function Distributions.fit(::Type{LastLinearBayesianCPD}, data::DataFrame, target::NodeName, parents::NodeNames)
    # m
    nparents = length(parents)
    m, ps = BayesianLinReg(nparents; update_prior=true, update_noise=true);
    # data
    X = data[!, parents] |> Array
    y = data[!,target] |> Vector
    # cpd
    cpd = LastLinearBayesianCPD(target, parents, m, ps)
    Distributions.fit!(cpd, X, y)
end

""" Use Distributions.fit interface
"""
function Distributions.fit(::Type{LastLinearBayesianCPD}, data::DataFrame, target::NodeName, pa::NodeNames, w_noise_scale::Float64)
    # m
    nparents = length(pa)
    m, ps = BayesianLinReg(nparents; update_prior=true, update_noise=true);
    # data
    X = data[!, pa] |> Array
    y = data[!,target] |> Vector
    # cpd
    cpd = LastLinearBayesianCPD(target, pa, m, ps)
    Distributions.fit!(cpd, X, y, w_noise_scale)
end

function Distributions.fit!(cpd::LastLinearBayesianCPD, X, y)
    n, d = size(X)
    @assert n > d
    @unpack m, ps = cpd
    cpd.ps = posterior(cpd.m, cpd.ps, X, y)
    return cpd
end

function Distributions.fit!(cpd::LastLinearBayesianCPD, X, y, w_noise_scale::Float64)
    @show w_noise_scale, name(cpd)
    cpd.ps = posterior(cpd.m, cpd.ps, X, y)
    s2 = cpd.ps[2]
    scaling = w_noise_scale^2
    eig = eigen(s2)
    Λ  = eig.values
    Λ .= max.(Λ, 0.0)
    Q = eig.vectors
    Λ .= max.(Λ, 0.0) .* scaling  # scale std of w to w_noise_scale
    cpd.ps[2] = Q * diagm(Λ) * Q'
    return cpd
end


"Maximum a posteriori"
function MAP(cpd::LastLinearBayesianCPD, data::DataFrame, target::NodeName, parents::NodeNames)
    X = data[!, parents] |> Array
    y = data[!,target] |> Vector
    MAP(cpd::LastLinearBayesianCPD, X, y)
end

function MAP(cpd::LastLinearBayesianCPD, X, y)
    ps = posterior(cpd.m, cpd.ps, X, y)
    return ps[1]
end

function test_linear_bayes_cpd()
    # data
    w = [-0.3, 0.5]
    n = 100
    X = rand(Uniform(-1, 1), n)
    X = hcat(X, ones(length(X)))
    ϵ = rand(Normal(0, 0.2), n)
    f(x, w) = x*w
    Y = f(X, w) + ϵ
    data = DataFrame(hcat(X, Y), [:X1, :X2, :Y])
    target = :Y
    pas = [:X1, :X2]  # parents
    cpd = Distributions.fit(LastLinearBayesianCPD, data, target, pas)
end

