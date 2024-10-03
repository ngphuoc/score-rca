using Flux: Data
using Graphs, BayesNets, Flux, PythonCall
using Parameters: @unpack
@unpack truncexpon, halfnorm = pyimport("scipy.stats")
@unpack random = pyimport("numpy")
include("lib/utils.jl")
include("lib/distributions.jl")
include("denoising-score-matching.jl")
include("bayesnets-extra.jl")
include("bayesnets-fit.jl")
include("data-rca.jl")

@info "#-- 0. load data"

function generate_linear_skewed(; args)
    g = BayesNet()
    ds = map((d, s) -> d(0, s), rand([Normal, Laplace], 2), rand(0.1:0.1:1, 2))
    d = MixtureModel(ds, [0.7, 0.3])
    cpd = RootCPD(:X1, d)
    push!(g, cpd)
    ε = rand(g, 300)
    #-- anomaly data
    da = MixtureModel(scale3.(d.components), d.prior)
    g.cpds[1].d = da
    εa = rand(g, 300)
    @≥ ε, εa Array.() transpose.()
    ds = [cpd.d for cpd in g.cpds]
    za = @> pval.(ds, eachrow(εa)) hcats transpose
    ya = 1 .- za
    # xs = -4:0.1:4
    # ys = pdf.(d, xs)
    # xa = εa[ya]
    # plot(xs, ys, lab="pdf")
    # scatter!(xa, fill(1e-2, length(xa)), markersize=3, lab="outlier")
    # savefig("fig/linear.png")
    nodes = names(g)
    anomaly_nodes = [1]
    x, xa = ε, εa

    X = x
    μx, σx = @> X mean(dims=2), std(dims=2);
    normalise_x(x) = @. (x - μx) / σx
    scale_ε(ε) = @. ε / σx
    @≥ X, x, xa normalise_x.();
    @≥ ε, εa scale_ε.();

    return g, nodes, x, ε, xa, εa, anomaly_nodes, μx, σx, ya
end

function get_ε_rankings(εa, ∇εa)
    @assert size(εa, 1) == d
    i = 1
    scores = Vector{Float64}[]  # 1 score vector for each outlier
    batchsize = size(εa, 2)
    for i = 1:batchsize
        tmp = Dict(j => ∇εa[j, i] * εa[j, i] for j = 1:d)
        ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]  # used
        score = zeros(d)
        for q in 1:max_k
            iq = findfirst(==(ranking[q]), 1:d)
            score[iq] = overall_max_k - q
        end
        push!(scores, score)
    end
    return scores
end

g, nodes, x, ε, xa, εa, anomaly_nodes, μx, σx, ya = generate_linear_skewed(; args)
n_anomaly_nodes = 1

fname = "results/random-linear-v2.csv"
rm(fname, force=true)

dfs = DataFrame(
               n_nodes = Int[],
               n_anodes = Int[],
               method = String[],
               noise_dist  = String[],
               data_id = Int[],
               ranking = Float64[],
               manual = Float64[],
               score = Float64[],
               k = Int[],
              )

include("method-siren.jl")

# include("method-bigen.jl")

# include("method-causalrca.jl")

# include("method-circa.jl")

# include("method-traversal.jl")

# dfs[!, :ndcg_manual] = round.(dfs[!, :ndcg_manual], digits=3)
# dfs[!, :ndcg_ranking] = round.(dfs[!, :ndcg_ranking], digits=3)

# CSV.write(fname, dfs, header=!isfile(fname), append=true)

