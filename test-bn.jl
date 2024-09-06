include("./data-rca.jl")
using BayesNets: forward

function BayesNets.forward(cpd::RootCPD, a::Assignment, sampler)
    noise = rand(cpd.d)
    μ = zero(noise)
    μ + noise, μ, noise
end

function BayesNets.forward(cpd::LocationCPD, a::Assignment, sampler)
    x = getindex.([a], cpd.parents)
    μ = only(cpd.μ(x))
    noise = rand(cpd.d)
    μ + noise, μ, noise
end

# A → C ← B
bn = BayesNet()
push!(bn, RootCPD(:a, Normal(0.0, 1.0)))
push!(bn, RootCPD(:b, Normal(0.0, 3.0)))
mlp = Dense(2=>1, bias=false)
push!(bn, LocationCPD(:c, [:a, :b], fmap(f64, mlp), Normal(0.0, 5)))

a, b, c = forward(bn, 5)

# Dict(:a=>1, :b=>2, :c=>1)

# t1 = forward(bn, 5)
# @test size(t1) == (5,3)
# @test t1[!,:a] == [1,1,1,1,1]
# @test t1[!,:b] == [2,2,2,2,2]

# t2 = forward(bn, 5, Assignment(:c=>1))
# @test size(t1) == (5,3)

# t3 = forward(bn, 5, :c=>1, :b=>2)
# @test size(t1) == (5,3)

# t4 = forward(bn, LikelihoodWeightedSampler(:c=>1), 5)
# # is there a test here?

# t5 = forward(bn, GibbsSampler(Assignment(:c=>1), burn_in=5), 5)
# @test t5[!,:a] == [1,1,1,1,1]
# @test t5[!,:b] == [2,2,2,2,2]

