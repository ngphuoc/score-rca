using Lux, Accessors
using Lux: Dense, Chain, Parallel, @set!, AbstractExplicitLayer, AbstractExplicitContainerLayer

# skip state input for small, stateless models
@inline (m::Lux.AbstractExplicitLayer)(x::AbstractArray, ps) = m(x, ps, NamedTuple())

@inline function apply(f::Symbol, x, model::AbstractExplicitContainerLayer, ps, st::NamedTuple)
    y, st_f = getfield(model, f)(x, getfield(ps, f), getfield(st, f))
    st = @set st[f] = st_f
    return y, st
end

# W*x, size(W) == (out_dim, in_dim)
function lecun_normal(dims...; gain=1.0f0)
  std = Float32(gain * 1 / √nfan(dims...)[2]) # fan_in
  return randn(Float32, dims...) .* std
end

# W*x, size(W) == (out_dim, in_dim)
function lecun_uniform(dims...; gain=1.0f0)
  return (2rand(Float32, dims...) .- 1) .* Float32(gain * 1 / √nfan(dims...)[2]) # fan_in
end


# W*x, size(W) == (out_dim, in_dim)
function sqrt_normal(dims...; gain=1.0f0)
  std = Float32(1 / √√nfan(dims...)[2]) # fan_in
  return randn(Float32, dims...) .* std
end

# W*x, size(W) == (out_dim, in_dim)
function sqrt_uniform(dims...; gain=1.0f0)
  return (2rand(Float32, dims...) .- 1) .* Float32(1 / √√nfan(dims...)[2]) # fan_in
end

nothing
