using Statistics, Distributions
using Parameters: @unpack

""" Compute the z-score, and tail probability outlier score.
    zscore(x) = |x - EX| / σ(X)
    outlier score (x) = -log P(|X - EX| >= |x - EX|)
"""
struct ZOutlierScore
    μ
    σ
end

function Distributions.fit(ZOutlierScore, X)
    μ, σ = mean(X), std(X)
    ZOutlierScore(μ, σ)
end

function (score::ZOutlierScore)(X)
    @unpack μ, σ = score
    standard_normal = Normal(0, 1)
    @. -logcdf(standard_normal, -abs(X - μ) / σ)
end

