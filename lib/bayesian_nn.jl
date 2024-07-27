include("utils.jl")

"Sample from C+(0, σ)"
function rand_halfcauchy(σ::AbstractArray{T}, s) where {T}
  P = rand(T, s)
  @. σ * tan(0.5f0π * P)
end

is_bayesian_layer(l::T) where T= :post_μ_w ∈ fieldnames(T)

bayesian_layers(m::T) where T = filter(is_bayesian_layer, m.m.layers)

set_mode_bayesian_layers!(m::T, mode::A) where {T,A} = set_mode!.(bayesian_layers(m), mode)

set_mode!(l::T, mode::A) where {T, A} = (l.mode = mode)

sum_kl_divergence(m, N) = sum(kl_divergence, bayesian_layers(m)) / N

