using Revise, BenchmarkTools
include("./data-rca.jl")
include("./bayesnets-extra.jl")

# A → C ← B
bn = BayesNet()
push!(bn, RootCPD(:a, Normal(0.0, 1.0)))
push!(bn, RootCPD(:b, Normal(0.0, 3.0)))
mlp = Dense(2=>1, bias=false)
push!(bn, LocationCPD(:c, [:a, :b], fmap(f64, mlp), Normal(0.0, 5)))

ε = sample_noise(bn, 10)
a = @> bn.dag adjacency_matrix Matrix{Bool}
ii = @> a eachcol findall.()
z = forwardv2(bn, ε, ii)

function ε_func(ε)
    @> forwardv2(bn, ε, ii) sum
end

ε′, = Zygote.gradient(ε_func, ε)
@btime Zygote.gradient(ε_func, ε);

