include("utils.jl")
using Functors

Relu(x::AbstractArray{T}) where {T} = relu.(x)

struct Reshape{S}
  shape::S
end

Reshape(s...) = Reshape{typeof(s)}(s)

(l::Reshape)(x) = reshape(x, l.shape)

function linear_decay(epoch_num, decay_start, total_epochs, start_value)
  if epoch_num < decay_start
    return start_value
  end
  return start_value * (total_epochs - epoch_num) / (total_epochs - decay_start)
end

mutable struct Bias{T}
  β::T
end

function Bias(sz::Integer...; initβ=i -> zeros(Float32, i))
  Bias(initβ(sz))
end

@functor Bias

(a::Bias)(x::AbstractArray) = x .+ a.β

function Base.show(io::IO, l::Bias)
  print(io, "Bias(", size(l.β), ")")
end

function flatten(x::AbstractArray{T,N}; keepdims::Integer) where {T,N}
  @assert N >= keepdims
  s = size(x)
  reshape(x, s[1:keepdims-1]..., prod(s[keepdims:end]))
end

function flatten(x::AbstractArray{T,N}, keepdims::Integer) where {T,N}
  @assert N >= keepdims
  s = size(x)
  reshape(x, s[1:keepdims-1]..., prod(s[keepdims:end]))
end

function NNlib.batched_mul(A::AbstractArray{T1,4}, B::AbstractArray{T2,4}) where {T1,T2}
  C = NNlib.batched_mul(flatten(A, 3), flatten(B, 3))
  s = size(A)
  reshape(C, s[1], :, s[3:end]...)
end

function NNlib.batched_mul(A::AbstractArray{T1,5}, B::AbstractArray{T2,5}) where {T1,T2}
  C = NNlib.batched_mul(flatten(A, 3), flatten(B, 3))
  s = size(A)
  reshape(C, s[1], :, s[3:end]...)
end

function NNlib.batched_mul(A::AbstractArray{T1,6}, B::AbstractArray{T2,6}) where {T1,T2}
  C = NNlib.batched_mul(flatten(A, 3), flatten(B, 3))
  s = size(A)
  reshape(C, s[1], :, s[3:end]...)
end

permutedims12(A::AbstractArray{T}) where {T} = permutedims(A, [2, 1, 3:ndims(A)...])

function btranspose(A::AbstractArray{T}) where {T}
  a
end

function uniform!(x, a, b)
  rand!(x) .* (b - a) .+ a
end

clip(x, l, u) = min.(max.(x, l), u)

function Constant(c)
    return (s...) -> c * ones(s...)
end

nothing
