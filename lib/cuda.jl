using EndpointRanges, ExponentialAction, LinearAlgebra, CUDA, Random
using ChainRulesCore: ignore_derivatives
using Lazy: take, repeatedly

function expm(A::CuArray{T, 2}, max_degree=2size(A, 1)) where T
  ExponentialAction.expv_taylor(1, A, I, max_degree)
end

function trace(A::Array{T, 2}) where T
  tr(A)
end

function trace(A::CuArray{T, 2}) where T
  sum(A .* (I + 0A))
end

using ChainRules, ChainRulesCore, Zygote
function ChainRulesCore.rrule(::typeof(trace), x::AbstractArray{T}) where {T}
  d = size(x, 1)
  function tr_pullback(ΔΩ)
    return (NoTangent(), Diagonal(fill!(similar(x, d), ΔΩ)))
  end
  return trace(x), tr_pullback
end

function ChainRulesCore.rrule(::typeof(expm), A0::StridedMatrix{T}) where {T}
  A = copy(A0)
  X, intermediates = _matfun!(exp, A)
  function exp_pullback(X̄)
    # Ensures ∂X is mutable. The outer `adjoint` is unwrapped without copy by
    # the default _matfun_frechet_adjoint!
    ΔX = unthunk(X̄)
    ∂X = ChainRulesCore.is_inplaceable_destination(ΔX) ? copy(ΔX) : convert(Matrix, ΔX')'
    ∂A = _matfun_frechet_adjoint!(exp, ∂X, A, X, intermediates)
    return NoTangent(), ∂A
  end
  return X, exp_pullback
end

function power(x::AbstractArray{T, 2}, n::Int) where T
    return if n == 1
        x
    elseif isodd(n)
        x * power(x*x, (n-1)÷2)
    else
        power(x*x, (n-1)÷2)
    end
end

