
cpd = MlpCPD(:Y, [:X], Dense(1, 1, relu), Normal(0f0, 1))
rand(cpd, :X => 1f0)
rand(cpd, :X => 1f0)
rand(cpd)

a = randn(100)
b = randn(100) .+ 2*a .+ 3

data = DataFrame(a=a, b=b)
cpdA = fit(StaticCPD{Normal}, data, :a)
cpdB = fit(LinearGaussianCPD, data, :b, [:a])
bn2 = BayesNet([cpdA, cpdB])
cpdB = LinearGaussianCPD(:b, [:a], [1.0], 0.0, 1.0)
cpdB(:a => 0.5)
df = rand(bn2, 100)
sort(df, [:a])

cpdB = MlpCPD(:b, [:a], Dense(1, 1), Normal(0.0, 1.0))
bn3 = BayesNet([cpdA, cpdB])
df = rand(bn2, 100)
sort(df, [:a])

cpdA = RootCPD(:a, [], Normal(0.0, 1.0))
cpdB = MlpCPD(:b, [:a], Dense(1, 1), Normal(0.0, 1.0))
bn = BayesNet([cpdA, cpdB])
df = rand(bn, 100)
sort(df, [:a])

@> scatter(eachcol(df)...) savefig("fig/test-mlp-cpd.png")

