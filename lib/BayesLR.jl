# using Statistics
using LinearAlgebra
using Measurements
using Printf
using LazyArrays

# model
σ_t = 0.2
β = (1/σ_t)^2
α = 2.0
σ_x = 1/√α
f(x, w) = x*w
prior = MvNormal(zeros(2), I(2))

function test_bayeslr()
    # data
    w = [-0.3, 0.5]
    n = 100
    X = rand(Uniform(-1, 1), n)
    X = hcat(X, ones(length(X)))
    ϵ = rand(Normal(0, 0.2), n)
    T = f(X, w) + ϵ
end

function log_evidence(w, x, t)
    @> -β/2 .* (t - x * w).^2 .- α/2 * w'w sum
end

mutable struct BayesLR
    α  # prior precision
    β

    weights
    iterationCount::Int
    done::Bool

    uncertaintyBasis::Vector{Measurement{Float64}}

    active::Vector{Int}
end

function BayesLR()
    Λ_0 = α*I(2)
    m_0 = [0.0, 0.0]
end

function update!(x, t)
    x = X[[i], :]
    t = T[[i]]
    w_llh = llh.(w_grid, [x], [t])
    Λ_n = Λ_0 + β .* x'x
    S_n = inv(Λ_n)
    m_n = S_n * (Λ_0 * m_0 + β .* x't)
    posterior = MvNormal(m_n, Symmetric(S_n))
    w_posterior = pdf.([posterior], w_grid)
    p1 = plot(heatmap(w_llh));
    p2 = plot(heatmap(w_posterior));
    scatter(X[1:i, 1], T[1:i], xlim=(-1, 1), ylim=(-1, 1));
    v = rand(posterior, 6)
    x_range = [-1:0.01:1...]
    y = x_range * v[[1], :] .+ v[[2], :]
    p3 = plot!(x_range, y);
    push!(plots, plot(p1, p2, p3))
    Base.show(plots[i])
    Λ_0 = Λ_n
    m_0 = m_n
end

function symmetric!(S)
    S .+= S'
    S ./= 2
end

###########################################################################################
# Hessian, Eq. 3.81

function hessian(eig, α, β)
    Q = eig.vectors
    D = Diagonal(α .+ β .* eig.values)

    H = Q * D * Q'
    symmetric!(H)

    return H
end

function hessian(m::BayesLR)
    α = m.α
    β = m.β

    return hessian(m.eig, α, β)
end


###########################################################################################
# Inverse Hessian

function hessianinv!(m::BayesLR)
    Λ = m.eig.values
    Q = m.eig.vectors

    α = m.α
    β = m.β

    D = Diagonal(inv.(α .+ β .* Λ))

    Hinv = m.Hinv
    Hinv .= @~ Q * D * Q'
    symmetric!(Hinv)

    return Hinv
end

###########################################################################################
# Log-determinant of Hessian

function logdetH(m::BayesLR)
    Λ = m.eig.values
    α = m.α
    β = m.β
    return logdetH(α, β, Λ)
end

function logdetH(α, β, Λ)
    sum(log, α .+ β .* Λ)
end



# Eq. 3.84
function updateWeights!(m::BayesLR)
    β = m.β
    Hinv = hessianinv!(m)

    m.weights .= @~ β .* (Hinv * m.Xty)
    return m.weights
end

function BayesLR(
          X::Matrix{T}
        , y::Vector{T}
        ; updatePrior=true
        , updateNoise=true
        ) where {T}

    (N, p) = size(X)

    XtX = X' * X
    Xty = X' * y
    yty = normSquared(y)

    symmetric!(XtX)

    return BayesLR(XtX, Xty, yty, N; updatePrior, updateNoise)
end

function BayesLR(
      XtX
    , Xty
    , yty
    , N
    ; updatePrior=true
    , updateNoise=true)
    eig = eigen(XtX)

    p = length(Xty)
    ps = [1:p;]

    XtX = view(XtX, ps, ps)
    Xty = view(Xty, ps)

    Λ  = eig.values
    Λ .= max.(Λ, 0.0)
    Q = eig.vectors

    α = 1.0
    β = 1.0

    ps = [1:p;]
    D = Diagonal(inv.(α .+ β .* Λ))

    Hinv = Q * D * Q'
    symmetric!(Hinv)
    Hinv = view(Hinv, ps, ps)

    weights = view(β .* (Hinv * Xty), ps)

    α0 = 1.0
    β0 = 1.0
    BayesLR(
        XtX
      , Xty
      , yty
      , N
      , Hinv
      , eig
      , α
      , updatePrior
      , β
      , updateNoise
      , weights
      , 0
      , false
      , zeros(p) .± 1.0
      , [1:p;],
      α0,
      β0
  )

end

"""
Sum of squared residuals. The norm-squared term in Eq. 3.82
"""
function ssr(m)
    XtX = m.XtX
    Xty = m.Xty
    yty = m.yty
    w = m.weights
    return yty - 2 * mydot(w, Xty) + w' * XtX * w
end

function Base.iterate(m::BayesLR{T}, iteration=1) where {T}
    m.done && return nothing

    N = m.N
    ps = m.active
    α = m.α
    β = m.β

    gamma(m) = effectiveNumParameters(m)

    XtX = m.XtX
    Xty = m.Xty
    yty = m.yty

    w = m.weights

    wᵀw = normSquared(w)
    rᵀr = ssr(m)
    if m.updatePrior
        m.α = gamma(m) / wᵀw  # Eq. 3.92
    end

    if m.updateNoise
        m.β = (N - gamma(m)) / rᵀr
    end

    updateWeights!(m)

    m.iterationCount += 1
    return (m, iteration + 1)
end

function fit!(m::BayesLR; kwargs...)
    m.done = false
    callback = get(kwargs, :callback, stopAtIteration(10))

    if m.updatePrior
        m.α = 1.0
    end

    if m.updateNoise
        m.β = 1.0
    end

    try
        for iter in m
            callback(iter)
        end
    catch e
        if e isa InterruptException
            @warn "Computation interrupted"
            return m
        else
            rethrow()
        end
    end
    return m
end

function logEvidence(m::BayesLR{T}) where {T}
    N = m.N
    α = m.α
    β = m.β
    rtr = ssr(m)
    return _logEv(N, rtr, m.active, α, β, m.eig.values, m.weights)
end

const log2π = log(2π)

# Eq. 3.86
function _logEv(N, rtr, active, α, β, Λ, w)
    p = length(active)

    logEv = 0.5 *
        ( p * log(α)
        + N * log(β)
        - (β * rtr + α * normSquared(w))  # Eq. 3.82
        - logdetH(α, β, Λ)
        - N * log2π
        )
    return logEv
end

# Eq. 3.91
function effectiveNumParameters(m::BayesLR)
    α_over_β = m.α / m.β

    Λ = m.eig.values

    return sum(λ -> λ / (α_over_β + λ), Λ)
end

function posteriorPrecision(m::BayesLR)
    return hessian(m)
end

posteriorVariance(m::BayesLR) = hessianinv!(m)

# Eq. ?
function posteriorWeights(m)
    p = length(m.active)
    ϕ = posteriorPrecision(m)
    U = cholesky!(ϕ).U

    w = m.weights
    ε = inv(U) * view(m.uncertaintyBasis, m.active)
    return w + ε
end

# TODO: Why is this slower than `posteriorWeights`?
function postWeights(m)
    α = m.α
    β = m.β
    Λ = m.Λ
    Vt = m.Vt
    V = Vt'

    w = view(m.weights, m.active)

    # S = V * diagm(sqrt.(inv.(α .+ β .* Λ))) * Vt
    # ε = S * view(m.uncertaintyBasis, m.active)

    ε = view(m.uncertaintyBasis, m.active)
    ε .= @~ Vt * ε
    ε ./= sqrt.(α .+ β .* Λ)
    ε .= @~ V * ε

    return w + ε
end


function predict(m::BayesLR, X; uncertainty=true, noise=false)
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

_predict(m, X, ::Val{false}, ::Val{false}) = X * m.weights

α(m::BayesLR) = m.α

priorVariance(m::BayesLR) = 1/m.α

priorScale(m::BayesLR) = sqrt(priorVariance(m))



β(m::BayesLR) = m.β

noiseVariance(m::BayesLR) = 1/m.β

noiseScale(m::BayesLR) = sqrt(noiseVariance(m))

function Base.show(io::IO, m::BayesLR{T}) where {T}
    @printf io "BayesLR model\n"
    @printf io "\n"
    @printf io "Log evidence: %3.2f\n" logEvidence(m)
    @printf io "Prior scale: %5.2f\n" priorScale(m)
    @printf io "Noise scale: %5.2f\n" noiseScale(m)
    @printf io "\n"
    @printf io "Coefficients:\n"
    weights = posteriorWeights(m)
    for (j,w) in zip(m.active, weights)
        @printf io "%3d: %5.2f ± %4.2f\n" j w.val w.err
    end
end

###########################################################################################
# Helpers

function normSquared(v::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(v)
        s += v[i]*v[i]
    end
    return s
end

mydot(x,y) = mydot(promote(x,y...)...)

function mydot(x::AbstractVector{T},y::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(x)
        s += x[i]*y[i]
    end
    return s
end

function update!(m)
    ps = m.active
    XtX = m.XtX = view(m.XtX.parent, ps, ps)
    Xty = m.Xty = view(m.Xty.parent, ps)
    weights = m.weights = view(m.weights.parent, ps)
    Hinv = m.Hinv = view(m.Hinv.parent, ps, ps)

    m.eig = eigen(collect(m.XtX))
    m.eig.values .= max.(m.eig.values, 0.0)

    α = m.α = m.α0
    β = m.β = m.β0

    # Eq. 3.54
    Q = m.eig.vectors
    Λ = m.eig.values
    D = Diagonal(inv.(α .+ β .* Λ))
    Hinv .= @~ Q * D * Q'

    m.weights .= @~ β .* (Hinv * Xty)  # Eq. 3.53
    return m
end

