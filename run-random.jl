using Graphs, BayesNets, Flux, PythonCall
using Parameters: @unpack
@unpack truncexpon, halfnorm = pyimport("scipy.stats")
include("lib/utils.jl")
include("lib/distributions.jl")
include("denoising-score-matching.jl")
include("bayesnets-extra.jl")
include("bayesnets-fit.jl")
include("data-rca.jl")

#-- 1. dag, linear model, and noises

@info "#-- 0. load data"

g, x, _, xa, _, _, _, ε, _, εa, _, _, anomaly_nodes = load_normalised_data(args);

fname = "results/random-graph-v2.csv"
rm(fname, force=true)

results = []

g, nodes, x, ε, xa, εa, anomaly_node, μx, σx = micro_service_data(; args);
include("method-siren.jl")

g, nodes, x, ε, xa, εa, anomaly_node, μx, σx = micro_service_data(; args);
include("method-bigen.jl")

g, nodes, x, ε, xa, εa, anomaly_node, μx, σx = micro_service_data(; args);
include("method-causalrca.jl")

g, nodes, x, ε, xa, εa, anomaly_node, μx, σx = micro_service_data(; args);
include("method-circa.jl")

g, nodes, x, ε, xa, εa, anomaly_node, μx, σx = micro_service_data(; args);
include("method-traversal.jl")

dfs[!, :ndcg_manual] = round.(dfs[!, :ndcg_manual], digits=3)
dfs[!, :ndcg_ranking] = round.(dfs[!, :ndcg_ranking], digits=3)

CSV.write(fname, dfs, header=!isfile(fname), append=true)

