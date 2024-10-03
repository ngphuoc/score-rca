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

function generate_scalar_skewed(; args)
    ds = map((d, s) -> d(0, s), rand([Normal, Laplace], 2), rand(0.1:0.1:1, 2))
    d = MixtureModel(ds, [0.7, 0.3])
    ε = rand(d, 300)
    #-- anomaly data
    da = MixtureModel(scale3.(d.components), d.prior)
    εa = rand(da, 300)
    za = pval(d, εa)
    ya = @> za .< 1e-3
    sum(ya)
    xs = -4:0.1:4
    ys = pdf.(d, xs)
    xa = εa[ya]
    plot(xs, ys, lab="pdf")
    scatter!(xa, fill(1e-2, length(xa)), markersize=3, lab="outlier")
    savefig("fig/scalar.png")
    return ε, εa
end

ε, εa = generate_scalar_skewed(; args)

fname = "results/random-scalar-v2.csv"
rm(fname, force=true)

dfs = DataFrame(
               n_nodes = Int[],
               n_anomaly_nodes = Int[],
               method = String[],
               noise_dist  = String[],
               data_id = Int[],
               ndcg_ranking = Float64[],
               ndcg_manual = Float64[],
               k = Int[],
              )

include("method-siren.jl")

include("method-bigen.jl")

include("method-causalrca.jl")

include("method-circa.jl")

include("method-traversal.jl")

dfs[!, :ndcg_manual] = round.(dfs[!, :ndcg_manual], digits=3)
dfs[!, :ndcg_ranking] = round.(dfs[!, :ndcg_ranking], digits=3)

CSV.write(fname, dfs, header=!isfile(fname), append=true)

