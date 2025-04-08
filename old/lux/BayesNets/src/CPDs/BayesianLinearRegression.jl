# using Statistics
using LinearAlgebra
using Measurements
using Printf
using LazyArrays
using Parameters: @unpack

export BayesianLinReg,
       get_lr_stats,
       get_univar_stats,
       posterior,
       distribution,
       effective_num_parameters,
       summary

const log2π = log(2π)

# inputs = (X, y)
# ps = (μ, Λ, α, β,)  # parameters

struct BayesianLinReg
    α0
    β0
    update_prior::Bool
    update_noise::Bool
end

function BayesianLinReg(d::Int; update_prior=true , update_noise=true, α0 = 1.0, β0 = 1.0)
    m = BayesianLinReg(α0, β0, update_prior, update_noise)
    # params
    α, β = α0, β0
    Λ = α*I(d)
    μ = zeros(d)
    ps = [μ, Λ, α, β]
    return m, ps
end

function get_univar_stats(x)
    x̄ = mean(x)
    s² = var(x)
    n = length(x)
    return (; x̄, s², n)
end

function get_lr_stats(X, y)
    (n, d) = size(X)
    XtX = X' * X
    Xty = X' * y
    yty = norm_squared(y)
    # XtX = XtX .* I(d)
    symmetric!(XtX)
    return (; XtX, Xty, yty, n)
end

# https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
function posterior(d::Normal, stats)
    @unpack x̄, s², n = stats
    μ0, σ0 = (d.μ, d.σ)
    σ0² = σ0^2
    sn² = 1 / (1/σ0² + n/s²)
    μn = sn² * (μ0/σ0² + n * x̄ / s²)
    return Normal(μn, √sn²)
end

"Return parameters of the posterior"
function posterior(m::BayesianLinReg, ps, X, y; kwargs...)
    stats = get_lr_stats(X, y)
    effective_num_parameters(m, ps, stats)
    posterior(m::BayesianLinReg, ps, stats; kwargs...)
end

function posterior(m::BayesianLinReg, ps, stats; niters=1)
    # γ(ps) = effective_num_parameters(m, ps, stats)
    γ = length(ps[1])  # use full number of parameters
    @unpack XtX, Xty, yty, n = stats
    for i = 1:niters
        (μ0, Λ0, α0, β0) = ps   # parameters
        α, β = α0, β0

        # update weight posterior
        Λ = Λ0 + β .* XtX
        S = inv(Λ)
        μ = S * (Λ0 * μ0 + β .* Xty)

        # update α, β
        ps = [μ, Λ, α, β]
        if m.update_prior
            w = μ
            wᵀw = norm_squared(w)
            # α = γ(ps) / wᵀw  # Eq. 3.92
            α = γ / wᵀw  # Eq. 3.92
        end
        if m.update_noise
            rᵀr = ssr(m, ps, stats)
            # β = (n - γ(ps)) / rᵀr  # Eq. 3.95
            β = (n - γ) / rᵀr  # Eq. 3.95
            # @show n, γ, rᵀr
        end
        ps = [μ, Λ, α, β]
    end
    return ps
end

function distribution(m::BayesianLinReg, ps)
    (μ, Λ, α, β) = ps   # parameters
    MvNormal(μ, Symmetric(inv(Λ)))
end

function symmetric!(S)
    S .+= S'
    S ./= 2
end

# Eq. 3.91
function effective_num_parameters(m::BayesianLinReg, ps, stats)
    (μ, Λ, α, β,) = ps   # parameters
    α_over_β = α / β
    eig = eigen(stats.XtX)
    return sum(λ -> λ / (α_over_β + λ), eig.values)
end

"""
Sum of squared residuals. The norm-squared term in Eq. 3.82
"""
function ssr(m, ps, stats)
    (w, Λ, α, β,) = ps   # parameters
    return stats.yty - 2 * mydot(w, stats.Xty) + w' * stats.XtX * w
end

function logdetH(m::BayesianLinReg)
    Λ = m.eig.values
    α = m.priorPrecision
    β = m.noisePrecision
    return logdetH(α, β, Λ)
end

function logdetH(α, β, Λ)
    sum(log, α .+ β .* Λ)
end

# Eq. 3.86
function log_evidence(m::BayesianLinReg, ps, stats)
    (w, Λ, α, β,) = ps   # parameters
    rtr = ssr(m, ps, stats)
    d = length(w)
    n = stats.n

    logEv = 0.5 * (
                   d * log(α)
                   + n * log(β)
                   - (β * rtr + α * norm_squared(w))  # Eq. 3.82
                   - logdetH(α, β, Λ)
                   - n * log2π
                  )
    return logEv
end

function predict(m::BayesianLinReg, X; uncertainty=true, noise=false)
    noise &= uncertainty
    # dispatch to avoid type instability
    return _predict(m, X, Val(uncertainty), Val(noise))
end

function _predict(m, X, ::Val{true}, ::Val{true})
    yhat = _predict(m, X, Val(true), Val(false))
    n = length(yhat)
    noise = zeros(n) .± noiseScale(m)
    yhat .+= noise
    return yhat
end

_predict(m, X, ::Val{true}, ::Val{false}) = X * posteriorWeights(m)

_predict(m, X, ::Val{false}, ::Val{true}) = @error "Noise requires uncertainty"

_predict(m, X, ::Val{false}, ::Val{false}) = X * m.μ

############ Helpers

function norm_squared(v::AbstractVector{Y}) where {Y}
    s = zero(Y)
    @inbounds @simd for i ∈ eachindex(v)
        s += v[i]*v[i]
    end
    return s
end

mydot(x,y) = mydot(promote(x,y...)...)

function mydot(x::AbstractVector{Y}, y::AbstractVector{Y}) where {Y}
    s = zero(Y)
    @inbounds @simd for i ∈ eachindex(x)
        s += x[i]*y[i]
    end
    return s
end

function summary(BayesianLinReg)
end
