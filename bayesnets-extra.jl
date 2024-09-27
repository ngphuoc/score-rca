using BayesNets, Parameters
include("bayesnets-fit.jl")

@showfields BayesNet
@showfields NonlinearGaussianCPD
@showfields NonlinearScoreCPD

""" Sample zero mean noises from cpd.d """
function sample_noise(cpd::CPD, n_samples::Int)
    # rand(cpd.d, n_samples)
    rand((-1f0, 1f0), n_samples) .* rand(cpd.d, n_samples)  # symmetric about 0
end

""" Sample zero mean noises from bn.cpds.d """
function sample_noise(bn::BayesNet, n_samples::Int)
    @> sample_noise.(bn.cpds, n_samples) transpose.() vcats
end

function forward_1step_mean(bn::BayesNet, x::AbstractMatrix{T}, ii) where T
    @> forward.(bn.cpds, getindex.([x], ii, :)) vcats
end

function forward_1step(g::BayesNet, x::AbstractMatrix{T}) where T
    d = nv(g.dag)
    @> forward_1step.(1:d; g, x) vcats
end

function forward_1step(j::Int; g, x)
    cpd = g.cpds[j]
    paj = @> g.dag adjacency_matrix Matrix{Bool} getindex(:, j)
    y = forward(cpd, x[paj, :])
end

function forward_1step_scaled(g::BayesNet, x::AbstractMatrix{T}, μx, σx) where T
    d = @> g.dag adjacency_matrix size(1)
    @> forward_1step_scaled.(1:d; g, x, μx, σx) vcats
end

function forward_1step_scaled(j::Int; g, x, μx, σx)
    cpd = g.cpds[j]
    paj = @> g.dag adjacency_matrix Matrix{Bool} getindex(:, j)
    parent_child(x) = x[paj, :], x[[j], :]
    xs, μs, σs = @> x, μx, σx parent_child.()
    y = forward_scaled(cpd, xs, μs, σs)
end

""" using input pairs to simplify args
"""
function forward_scaled(cpd, xs, μs, σs)
    x1 = @. xs[1] * σs[1] + μs[1]
    x2 = forward(cpd, x1)
    @. (x2 - μs[2]) / σs[2]
end

function forward_scaled(cpd, x, μx, σx, μy, σy)
    x0 = @. x * σx + μx
    y = forward(cpd, x0)
    @. (y - μy) / σy
end

function forward(cpd::RootCPD, x::AbstractArray{T}) where T
    μ = zero(similar(x, 1, size(x, 2)))
end

function forward(cpd::MlpCPD, x::AbstractArray{T}) where T
    μ = cpd.mlp(x)
end

function forward(cpd::LinearCPD, x::AbstractArray{T}) where T
    μ = cpd.a'x
end

function forward(cpd::LinearBayesianCPD, x::AbstractArray{T}) where T
    @unpack target, parents, m, ps, b = cpd
    μ, Λ, α, β = ps
    σ = 1 / β
    Σ = inv(Λ)
    w = rand(MvNormal(μ, Symmetric(Σ)))
    y = μ'x .+ b
    # y = w'x .+ b
    return y
end

function forward(cpd::LocationCPD, x::AbstractArray{T}) where T
    μ = cpd.μ(x)
end

""" Forward the noises throught the bn following the dag topo-order
TODO: more efficient implementation
"""
function forward(bn::BayesNet, ε::AbstractMatrix{T}) where T
    ii = @> bn.dag adjacency_matrix Matrix{Bool} eachcol findall.()
    forward(bn, ε, ii)
end

function forward(bn::BayesNet, ε::AbstractMatrix{T}, ii::Vector{Vector{Int64}}) where T
    d = length(bn.cpds)
    batchsize = size(ε, 2)
    X = zero(similar(ε, 0, batchsize))
    for j = 1:d
        y = zero(similar(ε, 1, batchsize))
        x = X[ii[j], :]
        if length(x) > 0
            y += forward(bn.cpds[j], x)
        end
        y += ε[[j], :]
        X = vcat(X, y)
    end
    X
end

# this version uses feature indices as input to avoid Zygote mutation error
function forward_leaf(bn::BayesNet, ε::AbstractMatrix{T}, ii) where T
    d = length(bn.cpds)
    batchsize = size(ε, 2)
    X = zero(similar(ε, 0, batchsize))
    for j = 1:d
        y = zero(similar(ε, 1, batchsize))
        x = X[ii[j], :]
        if length(x) > 0
            y += forward(bn.cpds[j], x)
        end
        y += ε[[j], :]
        X = vcat(X, y)
    end
    X[end, :]
end

# this version uses feature indices as input to avoid Zygote mutation error
function forward_leaf(bn::BayesNet, ε::AbstractMatrix{T}) where T
    d = length(bn.cpds)
    ii = @> bn.dag adjacency_matrix Matrix{Bool} eachcol findall.()
    batchsize = size(ε, 2)
    X = zero(similar(ε, 0, batchsize))
    for j = 1:d
        y = zero(similar(ε, 1, batchsize))
        x = X[ii[j], :]
        if length(x) > 0
            y += forward(bn.cpds[j], x)
        end
        y += ε[[j], :]
        X = vcat(X, y)
    end
    X[end, :]
end

