using EndpointRanges, LinearAlgebra, Random
using MacroTools: postwalk, striplines, isexpr, @forward, @capture, animals, striplines
using ChainRulesCore: ignore_derivatives
using Lazy: take, repeatedly
import SplittableRandoms
using SplittableRandoms: SplittableRandom
include("repeat.jl")
using Flux: onehot
import NNlib

"Fast zip+splat"
zips(a::Vector{T}) where {T} = map(t -> map(x -> x[t], a), 1:length(first(a)))

zips(a::Vector{<:Tuple}) = tuple(map(t -> map(x -> x[t], a), 1:length(first(a)))...)

zips(a::T) where {T<:Tuple} = map(t -> map(x -> x[t], a), 1:length(first(a)))

""" cat+splat
    julia> cats([rand(1, 2) for _=1:3], dims=1)
    3×2 Matrix{Float64}:
     0.701238  0.704747
     0.464978  0.769693
     0.800235  0.947824
"""
cats(xs; dims=ndims(xs[1]) + 1) = cat(xs..., dims=dims)
vcats(xs) = cat(xs..., dims=1)
hcats(xs) = cat(xs..., dims=2)

macro extract(m, vs)
  rhs = Expr(:tuple)
  for v in vs.args
    push!(rhs.args, :($m.$v))
  end
  ex = :($vs = $rhs) |> striplines
  esc(ex)
end

"""
Extends Lazy's @>
"""
macro >(exs...)
  @assert length(exs) > 0
  callex(head, f, x, xs...) = ex = :_ in xs ? Expr(:call, Expr(head, f, xs...), x) : Expr(head, f, x, xs...)
  thread(x) = isexpr(x, :block) ? thread(rmlines(x).args...) : x
  thread(x, ex) =
    if isexpr(ex, :call, :macrocall)
      callex(ex.head, ex.args[1], x, ex.args[2:end]...)
    elseif isexpr(ex, :tuple)
      Expr(:tuple,
        map(ex -> isexpr(ex, :call, :macrocall) ?
                  callex(ex.head, ex.args[1], x, ex.args[2:end]...) :
                  Expr(:call, ex, x), ex.args)...)
    elseif @capture(ex, f_.(xs__))
      :($f.($x, $(xs...)))
    elseif isexpr(ex, :block)
      thread(x, rmlines(ex).args...)
    else
      Expr(:call, ex, x)
    end
  thread(x, exs...) = reduce(thread, exs, init=x)
  esc(thread(exs...))
end

macro >₂(exs...)
    thread(x) = isexpr(x, :block) ? thread(rmlines(x).args...) : x
    thread(x, ex) =
    isexpr(ex, :macrocall)        ? Expr(ex.head, ex.args[1], ex.args[2:3], x, ex.args[4:end]...) :
    isexpr(ex, :call,)            ? Expr(ex.head, ex.args[1:2]..., x, ex.args[3:end]...) :
    @capture(ex, f_.(xs__))       ? :($f.($(xs[1]), $x, $(xs[2:end]...))) :
    isexpr(ex, :block)            ? thread(x, rmlines(ex).args...) :
    Expr(:call, ex, x)
    thread(x, exs...) = reduce(thread, exs, init=x)
    esc(thread(exs...))
end

macro >=(x, exs...)
  esc(macroexpand(Main, :($x = @> $x $(exs...))))
end

var"@≥" = var"@>="

macro >=₂(x, exs...)
  esc(macroexpand(Main, :($x = @>₂ $x $(exs...))))
end

var"@≥₂" = var"@>=₂"

const squeeze = dropdims

# accuracy(x, y, m) = mean(onecold(cpu(m(x))) .== onecold(cpu(y)))
#
# function accuracy(d, m)
#   ŷ, y = @> map(d) do (x, y)
#     onecold(cpu(m(x))), onecold(cpu(y))
#   end zips cats.(dims=1)
#   mean(ŷ .== y)
# end
#
function copystruct!(a::T, b::U) where {T,U}
  for f in fieldnames(T)
    setfield!(a, f, getfield(b, f))
  end
  b
end

call(f, x) = f(x)

function vecnorm(x; dims=1)
  .√sum(abs2, x, dims=dims)
end

function expanddims(a::AbstractArray{T1,N}, dims::NTuple{N,Integer}) where {T1,N}
  a .+ oftype(a, zeros(dims))
end

function expanddims(a::AbstractArray{T1,N}, dim::Int, factor::Int) where {T1,N}
  s = size(a)
  a = @> flatten(a, dim + 1) flatten unsqueeze(2)
  a = a .+ oftype(a, zeros(1, factor, 1))
  s = (s[1:dim-1]..., s[dim] * factor, s[dim+1:end]...)
  a = reshape(a, s)
end

"Expand a as b in dims"
function expandas(a::AbstractArray{T1,N}, b::AbstractArray{T2,N}, dims::T3) where {T1,T2,T3,N}
  s = (size(a)[1:dims[1]-1]..., size(b)[dims]...)
  r = a .+ oftype(a, zeros(s...))
end

function expandboth(ab::Tuple{AbstractArray{T1,N},AbstractArray{T2,N}}, dims::T3=1:ndims(ab[1])) where {T1,T2,T3,N}
  a, b = ab
  s = max.(size(a), size(b))
  s = (fill(1, dims[1] - 1)..., s[dims]...)
  a = a .+ oftype(a, zeros(s...))
  b = b .+ oftype(b, zeros(s...))
  a, b
end

squeezeall(a::AbstractArray{T,N}) where {T,N} = reshape(a, (filter(x -> x != 1, size(a))...,))

function πs(rng, dim, xs...)
  if dim > 0
    Ipre = [Colon() for _ = 1:dim-1]
    map(xs) do x
      x[Ipre..., rng, [Colon() for _ = dim+1:ndims(x)]...]
    end
  else
    dim = abs(dim)
    Ipre = [Colon() for _ = 1:dim-1]
    map(xs) do x
      x[reverse([Ipre..., rng, [Colon() for _ = dim+1:ndims(x)]...])...]
    end
  end
end

π₃(rng, xs...) = πs(rng, 3, xs...)
π₂(rng, xs...) = πs(rng, 2, xs...)
π₁(rng, xs...) = πs(rng, 1, xs...)
π₋₁(rng, xs...) = πs(rng, -1, xs...)
π₋₂(rng, xs...) = πs(rng, -2, xs...)
π₋₃(rng, xs...) = πs(rng, -3, xs...)

π₃(x; i=1:size(x, 3)) = πs(i, 3, x)[1]
π₂(x; i=1:size(x, 2)) = πs(i, 2, x)[1]
π₁(x; i=1:size(x, 1)) = πs(i, 1, x)[1]
π₋₁(x; i=1:size(x)[end]) = πs(i, -1, x)[1]
π₋₂(x; i=1:size(x)[end-1]) = πs(i, -2, x)[1]
π₋₃(x; i=1:size(x)[end-2]) = πs(i, -3, x)[1]

# todo: cartesian index
function macro_π(rng, dim, xs...)
  ex = quote end
  #= xs = esc.(xs) =#
  i = gensym()
  #= push!(ex.args, :($i = squeezeall($rng))) =#
  push!(ex.args, :($i = $rng))
  if dim > 0
    Ipre = [Colon() for _ = 1:dim-1]
    foreach(x -> push!(ex.args, :($x = $x[$Ipre..., $i, [Colon() for _ = $dim+1:length(size($x))]...])), xs)
  else
    dim = abs(dim)
    Ipre = [Colon() for _ = 1:dim-1]
    foreach(x -> push!(ex.args, :($x = $x[reverse([$Ipre..., $i, [Colon() for _ = $dim+1:length(size($x))]...])...])), xs)
  end
  push!(ex.args, length(xs) > 1 ? Expr(:tuple, xs...) : xs[1])
  # @show ex
  esc(ex)
end

function Base.selectdim(xs::Tuple, dim::Union{EndpointRanges.IBegin,EndpointRanges.IEnd,EndpointRanges.IndexFunction}, rng::R) where {R}
  map(xs) do x
    selectdim(x, dim, rng)
  end
end

function Base.selectdim(x::AbstractArray, dim::Union{EndpointRanges.IBegin,EndpointRanges.IEnd,EndpointRanges.IndexFunction}, rng::R) where {R}
  Base.selectdim(x, dim(1:ndims(x)), rng)
end

# a = rand(2,3,4);
# b = rand(3, 4, 5, 6)
# @> selectdim((a, b), iend, 1:2) size.()
# @> selectdim((a, b), iend-1, 1:2) size.()
# @≥ a, b selectdim(iend-1, 1:2)
# @> a, b size.()

macro π₃(rng, xs...)
  macro_π(rng, 3, xs...)
end

macro π₂(rng, xs...)
  macro_π(rng, 2, xs...)
end

macro π₁(rng, xs...)
  macro_π(rng, 1, xs...)
end

macro π(rng, xs...)
  macro_π(rng, 1, xs...)
end

macro π₋₁(rng, xs...)
  macro_π(rng, -1, xs...)
end

macro π₋₂(rng, xs...)
  macro_π(rng, -1, xs...)
end

macro π₋₃(rng, xs...)
  macro_π(rng, -1, xs...)
end

using LinearAlgebra
using ExponentialAction

joinexp(xs::AbstractArray) = joinexp(xs...)
joinexp(xs...) = Expr(:block, xs...)

macro env(sym, default)
  default = Core.eval(Main, default)
  t = typeof(default)
  v = get(ENV, string(sym), "$default")
  if t ≠ String
    v = Meta.parse(v)
  end
  esc(:($sym = $v))
end

macro env(ex)
  nt = []
  function f(sym, default)
    default = Core.eval(Main, default)
    t = typeof(default)
    dic = "_querystr" in keys(ENV) ? fromquerystr(ENV["_querystr"]) : ENV
    v = default
    if string(sym) in keys(dic)
      v = dic[string(sym)]
      if default isa Function
        v = Meta.parse(v) |> eval
      elseif default isa String
        v = v
      elseif default isa DataType
        v = Core.eval(Main, Meta.parse(v))
      else
        v = convert(t, Meta.parse(v) |> eval)
      end
    end
    push!(nt, (sym, v))
    esc(:($sym = $v))
  end
  function makenamedtuple(nt)
    ex = Expr(:tuple)
    map(nt) do t
      push!(ex.args, Expr(:(=), t...))
    end
    esc(ex)
  end
  ex = striplines(ex)
  ss = [x.args[1] for x ∈ ex.args]
  ex = joinexp([f(x.args...) for x ∈ ex.args]..., makenamedtuple(nt))
  # push!(ex.args, esc(:(@astruct $(ss...))))
  # @show ex
  ex
end

function πs(rng, dim, xs...)
  if dim > 0
    Ipre = [Colon() for _ = 1:dim-1]
    map(xs) do x
      x[Ipre..., rng, [Colon() for _ = dim+1:ndims(x)]...]
    end
  else
    dim = abs(dim)
    Ipre = [Colon() for _ = 1:dim-1]
    map(xs) do x
      x[reverse([Ipre..., rng, [Colon() for _ = dim+1:ndims(x)]...])...]
    end
  end
end

π₃(rng, xs...) = πs(rng, 3, xs...)
π₂(rng, xs...) = πs(rng, 2, xs...)
π₁(rng, xs...) = πs(rng, 1, xs...)
π₋₁(rng, xs...) = πs(rng, -1, xs...)
π₋₂(rng, xs...) = πs(rng, -2, xs...)
π₋₃(rng, xs...) = πs(rng, -3, xs...)

π₃(x; i=1:size(x, 3)) = πs(i, 3, x)[1]
π₂(x; i=1:size(x, 2)) = πs(i, 2, x)[1]
π₁(x; i=1:size(x, 1)) = πs(i, 1, x)[1]
π₋₁(x; i=1:size(x)[end]) = πs(i, -1, x)[1]
π₋₂(x; i=1:size(x)[end-1]) = πs(i, -2, x)[1]
π₋₃(x; i=1:size(x)[end-2]) = πs(i, -3, x)[1]

Base.show(io::IO, ::MIME"text/plain", a::AbstractArray{T,3}) where {T} =
  println(io, typeof(a), " size ", size(a))
Base.show(io::IO, ::MIME"text/plain", a::AbstractArray{T,4}) where {T} =
  println(io, typeof(a), " size ", size(a))
Base.show(io::IO, ::MIME"text/plain", a::AbstractArray{T,5}) where {T} =
  println(io, typeof(a), " size ", size(a))
Base.show(io::IO, ::MIME"text/plain", a::AbstractArray{T,6}) where {T} =
  println(io, typeof(a), " size ", size(a))

# ref https://discourse.julialang.org/t/does-julia-have-any-utility-to-split-rngs-for-later-use-in-parallel/82775/4
function rng_split(rng::Xoshiro, N)
    map(Random.XoshiroSimd.forkRand(rng, Val(N))...) do si...
        Xoshiro((s.value for s in si)...)
    end
end

@inline function Base.split(rng::SplittableRandom, n::Int)
    take(n, repeatedly(() -> SplittableRandoms.split(rng)))
end

# https://github.com/denizyuret/Knet.jl/blob/master/src/ops20/conv.jl#L222
mat(x; dims::Int=ndims(x)-1)=reshape(x, (dims > 0 ? prod(size(x,i) for i in 1:dims) : 1), :)

average(xs...) = +(xs...)/length(xs)

function binary_vec(words, vocab)
    mapreduce(Base.Fix2(onehot, vocab), +, words)
end

function namedtuple_dict(; kw...)
    # Dict(pairs((; kw...)))
    Dict(string(k) => v for (k, v) in pairs(kw))
end

nothing
