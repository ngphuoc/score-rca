include("data-rca.jl")

a = randn(100)
b = randn(100) .+ 2*a .+ 3
# data = DataFrame(a=a, b=b)
data = vcat(a', b')

cpdA = RootCPD(:A, [], Normal(0f0, 1))

cpdB = LinearCPD(:B, [:A], randn(1), Normal(0f0, 1))

Distributions.fit!(cpdA, data, 1, Int[])
Distributions.fit!(cpdB, data, 2, [1])

bn = BayesNet([cpdA, cpdB])
@> bn.dag adjacency_matrix

Distributions.fit!(bn, data)

# cpdB = LinearGaussianCPD(:b, [:a], [1.0], 0.0, 1.0)
# cpdB(:a => 0.5)
# df = rand(bn2, 100)
# sort(df, [:a])

# cpdB = LinearCPD(:b, [:a], Dense(1, 1), Normal(0.0, 1.0))
# bn3 = BayesNet([cpdA, cpdB])
# df = rand(bn2, 100)
# sort(df, [:a])

# cpdA = RootCPD(:a, [], Normal(0.0, 1.0))
# cpdB = LinearCPD(:b, [:a], Dense(1, 1), Normal(0.0, 1.0))
# bn = BayesNet([cpdA, cpdB])
# df = rand(bn, 100)
# sort(df, [:a])

# @> scatter(eachcol(df)...) savefig("fig/test-mlp-cpd.png")

