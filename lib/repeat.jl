function _repeat(x::AbstractArray, counts::Integer...)
    N = max(ndims(x), length(counts))
    size_y = ntuple(d -> size(x,d) * get(counts, d, 1), N)
    size_x2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : 1, 2*N)

    ## version without mutation
    # ignores = ntuple(d -> reshape(Base.OneTo(counts[d]), ntuple(_->1, 2d-1)..., :), length(counts))
    # y = reshape(broadcast(first∘tuple, reshape(x, size_x2), ignores...), size_y)

    # ## version with mutation
    size_y2 = ntuple(d -> isodd(d) ? size(x, 1+d÷2) : get(counts, d÷2, 1), 2*N)
    y = similar(x, size_y)
    reshape(y, size_y2) .= reshape(x, size_x2)
    y
end

function _repeat(x::AbstractArray; inner=1, outer=1)
    N = max(ndims(x), length(inner), length(outer))
    size_y = ntuple(d -> size(x, d) * get(inner, d, 1) * get(outer, d, 1), N)
    size_y3 = ntuple(3*N) do d3
        dim, class = divrem(d3+2, 3)  # e.g. for x::Matrix, [divrem(n+2,3) for n in 1:3*2]
        class == 0 && return get(inner, dim, 1)
        class == 1 && return size(x, dim)
        class == 2 && return get(outer, dim,1)
    end
    size_x3 = ntuple(3*N) do d3
        dim, class = divrem(d3+2, 3)
        class == 1 ? size(x, dim) : 1
    end
    y = similar(x, size_y)
    reshape(y, size_y3) .= reshape(x, size_x3)
    y
end

# _get(t::Tuple, i::Int, default) = i in 1:length(t) ? t[i] : default
# _get(x::Number, i::Int, default) = i==1 ? x : default
# These are in Compat.jl now

using Zygote, ChainRules
using Zygote: unthunk
using ChainRules: ZeroTangent

function ChainRules.rrule(::typeof(_repeat), x::AbstractArray, counts::Integer...)
    size_x = size(x)
    function repeat_pullback_1(dy_raw)
        dy = unthunk(dy_raw)
        size2ndims = ntuple(d -> isodd(d) ? get(size_x, 1+d÷2, 1) : get(counts, d÷2, 1), 2*ndims(dy))
        reduced = sum(reshape(dy, size2ndims); dims = ntuple(d -> 2d, ndims(dy)))
        return (ZeroTangent(), reshape(reduced, size_x))
    end
    return repeat(x, counts...), repeat_pullback_1
end

function ChainRules.rrule(::typeof(_repeat), x::AbstractArray{T,N}; inner=ntuple(_->1,N), outer=ntuple(_->1,N)) where {T,N}
    size_x = size(x)
    function repeat_pullback_2(dy_raw)
        dy = unthunk(dy_raw)
        size3ndims = ntuple(3*ndims(dy)) do d3
            dim, class = divrem(d3+2, 3)  # e.g. for x::Matrix, [divrem(n+2,3) for n in 1:3*2]
            class == 0 && return get(inner, dim, 1)
            class == 1 && return size(x, dim)
            class == 2 && return get(outer, dim,1)
        end
        reduced = sum(reshape(dy, size3ndims); dims = ntuple(d -> isodd(d) ? 3(d÷2)+1 : 3(d÷2), 2*ndims(dy)))
        return (ZeroTangent(), reshape(reduced, size_x))
    end
    return repeat(x; inner=inner, outer=outer), repeat_pullback_2
end

