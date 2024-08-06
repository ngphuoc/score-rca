using Random, Combinatorics

rotate2d(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

function rotate!(X, rotations)
    d = size(X,1)
    # form random rotations
    drot = [ds for ds in keys(rotations) if ds.first == ds.second]
    if length(drot)>0
        p = first(drot)
        θ = rotations[p]
        rotations = if p.first > 0
            dims = ((i,j) for i in 1:d, j in 1:d if i != j) |> collect
            [(shuffle!(dims); first(dims)) => rand()*θ for i in 1:p.first]
        else
            dims = collect(1:d)
            [tuple(p...) => rand()*θ for p in combinations(dims,2)]
        end
    end
    # rotate dataset
    for ((i,j),θ) in rotations
        v = view(X, [i, j], :)
        v .= rotate2d(θ)*v
    end
end

"""
    nmoons(T, m, c; kwargs...) -> (Matrix{T}, Vector{Int})

Generate a half-moon dataset.

Parameters:
- `T`: data type for points' coordinates
- `m`: number of samples in the subset
- `c`: number of half-circle subsets in the generated dataset

Keyword parameters:
- `d`: dimension of the full space
- `r`: radius of the half-circle
- `ε`: variance of a random Gaussian noise
- `repulse::Tuple{T,T}`: half-circle subsets translation parameters
- `translation::Vector{T}`: dataset translation vector
- `rotations::Dict{Pair{Int,Int},T}`: dataset rotation parameters. Each rotation is specified as angle between particular pair of axes
- `shuffle`: perform point shuffling in the dataset, default value `true`
- `seed`: RNG seed value (if it's `nothing` then RNG will not be initialized), default value `nothing`

If the `rotations` specified as collection of `p => q => θ` pairs such that from axis of the dimension `p` rotation is perfomed
towards to the axis of dimension `q` on a radian value `θ`.
- if `p = q != 0` then `p` rotations between arbitrary pair of axis performed with angle chosen uniformly from range [0, θ].
- if `p = q = 0` then `d(d-1)/2` arbitrary rotations performed with angle chosen uniformly from range [0, θ].

Here is an example of dataset generation:
```jldoctest
julia> X, L = nmoons(Float64, 100, 2,      # Generate 200 Float64 points divided on two susbsets
                     ε=0.0, d=3,           # in 3D space with no noise
                     repulse=(0.25,0.0),   # repulse half-spheres in x-direction by r/4
                     translation=[0.25;0], # translated from origin
                     rotations=Dict((1=>3)=>π/3, (1=>2)=>-π/3)); # rotated 30° from X to Z, -30° from X to Y

julia> X
3×200 Array{Float64,2}:
 0.125      0.179973   0.233692  …   0.262203   0.319      0.375
-0.216506  -0.183582  -0.149013     -0.709903  -0.680664  -0.649519
 0.433013   0.431233   0.425903      1.29193    1.29726    1.29904
```
"""
function nmoons(::Type{T}, m::Int=100, c::Int=2;
                shuffle::Bool=false, ε::Real=0.1, d::Int = 2, r::Real=1.0,
                repulse::Tuple{T,T}=(zero(T),zero(T)),
                translation::Vector{T}=fill(zero(T),d),
                rotations::Dict{Pair{Int,Int},T} = Dict{Pair{Int,Int},T}(),
                seed::Union{Integer,Nothing}=nothing) where {T <: Real}
    @assert d > 1 "The ambient dimention must be grater than 1"
    @assert length(translation) == d "The dimension of the translation vector must be $d"

    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)
    n = c*m
    ssizes = fill(m, c)
    ssizes[end] += n - m*c
    @assert sum(ssizes) == n "Incorrect partitioning"
    pi = convert(T, π)
    X = zeros(d,0)
    for (i, s) in enumerate(ssizes)
        pts = range(zero(T), pi, length=s)
        circ_x = r.*(cos.(pts).-1.0)
        circ_y = r.*sin.(pts)
        R = rotate2d(-(i-1)*(2*pi/c))
        dir=R*[-2,1.0]
        dir ./= abs(sum(dir))   # normalize directions
        C = R * hcat(circ_x, circ_y)'
        @debug "Half-circle $i"  dir="$dir"
        translate = dir.*r.*collect(repulse)
        @debug "Repulse $i"  translate="$translate"
        C = vcat(C .+ translate, zeros(d-2, s)) # translate & pad coordinates with 0s
        X = hcat(X, C)
    end
    # generate labels
    y = vcat([fill(i,s) for (i,s) in enumerate(ssizes)]...)
    # shuffle points
    if shuffle
        idx = randperm(rng, n)
        X, y = X[:, idx], y[idx]
    end
    # add noise to the dataset
    if ε > 0.0
        X += randn(rng, size(X)).*convert(T,ε/d)
    end
    # rotate dataset
    rotate!(X, rotations)
    # translate dataset
    X .+= translation
    return X, y
end

