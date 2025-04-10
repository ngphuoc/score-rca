using Graphs, BayesNets, Flux, PythonCall
using Parameters: @unpack
@unpack truncexpon, halfnorm = pyimport("scipy.stats")
include("lib/utils.jl")
include("lib/distributions.jl")
include("dsm-model.jl")
include("bayesnets-extra.jl")
include("bayesnets-fit.jl")
include("data-rca.jl")

results = []

g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, fpath = load_data(args; dir="datals");
include("method-siren.jl")

g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, fpath = load_data(args; dir="datals");
include("method-bigen.jl")

g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, fpath = load_data(args; dir="datals");
include("method-causalrca.jl")

g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, fpath = load_data(args; dir="datals");
include("method-circa.jl")

g, z, x, l, s, z3, x3, l3, s3, za, xa, la, sa, anomaly_nodes, fpath = load_data(args; dir="datals");
include("method-traversal.jl")

df = DataFrame(results)
fname = "results/ls.csv"
rm(fname, force=true)
CSV.write(fname, df, header=!isfile(fname), append=true)

