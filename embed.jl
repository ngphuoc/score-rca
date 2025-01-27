using Flux

# Define the sinusoidal embedding
struct SinusoidalEmbedding
    size::Int
    scale::Float64
end

function (e::SinusoidalEmbedding)(x::AbstractVector{<:Real})
    x = x .* e.scale
    half_size = div(e.size, 2)
    emb = log(1e4) ./ (half_size .- 1)
    emb = exp.(-emb .* (0:half_size-1))
    emb = x .* emb'
    return hcat(sin.(emb), cos.(emb))
end

# Define the linear embedding
struct LinearEmbedding
    size::Int
    scale::Float64
end

function (e::LinearEmbedding)(x::AbstractVector{<:Real})
    return x .* e.scale ./ e.size
end

# Define the learnable embedding
struct LearnableEmbedding
    size::Int
    linear::Dense
end

LearnableEmbedding(size::Int) = LearnableEmbedding(size, Dense(1, size))

function (e::LearnableEmbedding)(x::AbstractVector{<:Real})
    return e.linear(x ./ e.size)
end

# Define the identity embedding
struct IdentityEmbedding
end

function (e::IdentityEmbedding)(x::AbstractVector{<:Real})
    return x
end

# Define the zero embedding
struct ZeroEmbedding
end

function (e::ZeroEmbedding)(x::AbstractVector{<:Real})
    return zeros(length(x))
end

# Define the positional embedding
struct PositionalEmbedding
    layer::Union{SinusoidalEmbedding, LinearEmbedding, LearnableEmbedding, IdentityEmbedding, ZeroEmbedding}
end

function PositionalEmbedding(size::Int, type::String; scale=1.0)
    if type == "sinusoidal"
        return PositionalEmbedding(SinusoidalEmbedding(size, scale))
    elseif type == "linear"
        return PositionalEmbedding(LinearEmbedding(size, scale))
    elseif type == "learnable"
        return PositionalEmbedding(LearnableEmbedding(size))
    elseif type == "zero"
        return PositionalEmbedding(ZeroEmbedding())
    elseif type == "identity"
        return PositionalEmbedding(IdentityEmbedding())
    else
        throw(ArgumentError("Unknown positional embedding type: $type"))
    end
end

function (p::PositionalEmbedding)(x::AbstractVector{<:Real})
    return p.layer(x)
end