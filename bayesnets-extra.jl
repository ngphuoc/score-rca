using BayesNets
include("bayesnets-fit.jl")

@showfields BayesNet
@showfields NonlinearGaussianCPD
@showfields NonlinearScoreCPD

""" Sample zero mean noises from cpd.d """
function sample_noise(cpd::CPD, n_samples::Int)
    rand(cpd.d, n_samples)
end

""" Sample zero mean noises from bn.cpds.d """
function sample_noise(bn::BayesNet, n_samples::Int)
    @> sample_noise.(bn.cpds, n_samples) transpose.() vcats
end

function forward(cpd::RootCPD, x::AbstractArray{T}) where T
    μ = zero(similar(x, 1, size(x, 2)))
end

function forward(cpd::MlpCPD, x::AbstractArray{T}) where T
    μ = cpd.mlp(x)
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

function forward_1step_mean(bn::BayesNet, x::AbstractMatrix{T}, ii) where T
    @> forward.(bn.cpds, getindex.([x], ii, :)) vcats
end

