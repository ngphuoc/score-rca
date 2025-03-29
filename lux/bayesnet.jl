using Lux

Lux.Dense

abstract type AbstractBijector end

@concrete struct AffineBijector <: AbstractBijector
    shift <: AbstractArray
    log_scale <: AbstractArray
end

function AffineBijector(shift_and_log_scale::AbstractArray{T, N}) where {T, N}
    n = size(shift_and_log_scale, 1) ÷ 2
    idxs = ntuple(Returns(Colon()), N - 1)
    return AffineBijector(
        shift_and_log_scale[1:n, idxs...], shift_and_log_scale[(n + 1):end, idxs...]
    )
end

function forward_and_log_det(bj::AffineBijector, x::AbstractArray)
    y = x .* exp.(bj.log_scale) .+ bj.shift
    return y, bj.log_scale
end


