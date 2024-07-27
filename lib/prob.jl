include("utils.jl")

""" Bernoulli log pdf of data y given param θ."""
@inline function bernoulli_logpdf(y::AbstractArray{T}; θ::AbstractArray{T}) where {T}
    ϵ = T(1e-9)
    _1 = T(1)
    sum(@. y * log(θ + ϵ) + (_1 - y) * log(_1 - θ + ϵ))
end

""" Gaussian log pdf of data x given param θ."""
@inline function gaussian_logpdf(x::AbstractArray{T,N}; μ::AbstractArray{T,M}, ρ::AbstractArray{T,M}) where {T, M, N}
    σ² = exp.(ρ)
    half = T(0.5)
    log2π = T(log(2π))
    -half * sum(@. log2π + ρ + (x - μ)^2 / σ²)
end

""" KL divergence from a diagonal Gaussian to the standard Gaussian.
Input: μ, and ρ = logσ²
"""
@inline kl_standard((μ, ρ)) = kl_standard(μ, ρ)

@inline function kl_standard(μ::AbstractArray{T}, ρ::AbstractArray{T}) where {T}
    half = T(0.5)
    _1 = T(1)
    -half * sum(@. _1 + ρ - μ^2 - exp(ρ))
end

""" KL divergence from a diagonal Gaussian to the standard Gaussian.
Input: μ₁, ρ₁ = logσ₁², ...
"""
@inline function klqp((μ₁, ρ₁), (μ₂, ρ₂))
    half = T(0.5)
    _1 = T(1)
    -half * sum(@. _1 + (ρ₁ - ρ₂) - (μ₁ - μ₂)^2 / exp(ρ₂) - exp(ρ₁ - ρ₂))
end

# gaussian_sample(μ::T, σ::AbstractArray{T}) where {T} = μ .+ oftype(σ, randn(Float32, size(σ))) .* σ

# gaussian_sample(μ::AbstractArray{T}, σ::T) where {T} = μ .+ oftype(μ, randn(Float32, size(μ))) .* σ

@inline function gaussian_sample(rng::AbstractRNG, μ::AbstractArray{T}, σ) where {T}
    μ .+ oftype(μ, randn(rng, size(μ))) .* σ
end

"Sample from C+(0, σ)"
function rand_halfcauchy(σ::AbstractArray{T}, s) where {T}
  P = rand(T, s)
  @. σ * tan(0.5f0π * P)
end

"""Sample from Gumbel(0,1)
"""
samplegumbel(size, ϵ=eps(Float32)) = -log.(-log.(rand(size...) .+ ϵ) .+ ϵ)

"""Sample from Gumbel softmax: gumbelsoftmax(logπ, τ=1; dims=1, hard=true)
"""
function gumbelsoftmax(logπ, τ=1; dims=1, hard=true)
  y = softmax((logπ .+ samplegumbel(size(logπ))) ./ τ, dims=dims)
  if hard
    yhard = y .== maximum(y, dims=dims)
    y = ignore_derivatives(yhard .- y) .+ y
  end
  y
end

nothing
