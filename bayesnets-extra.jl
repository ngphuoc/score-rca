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

""" Forward the noises throught the bn following the dag topo-order
TODO: more efficient implementation
"""
function forwardv2(bn::BayesNet, ε::AbstractMatrix{T}) where T
    d = length(bn.cpds)
    X = fill!(similar(ε, 0, size(ε, 2)), 0)
    for j = 1:d
        # ε[i, :] .+= bn(ε)[i, :]
        y = zero(ε)
        x = X[ii[j], :]
        if length(x) > 0
            y += bn.cpds[j](x)
        end
        y += ε[j, :]
        X = vcat(X, y)
    end
    X[end, :]
end

