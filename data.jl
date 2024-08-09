include("./lib/utils.jl")
include("./lib/nn.jl")
include("./lib/nnlib.jl")
include("./lib/graph.jl")
include("./dlsm-losses.jl")
using Flux
using Flux: crossentropy
using Flux.Data: DataLoader
using DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2
using Plots

args = @env begin
    seed = 1
    to_device = gpu
end

function get_data()
    # create dag
    d = 2
    v = [:X, :Y]
    g = DiGraph(d)
    add_edge!(g, 1, 2);
    B = adjacency_matrix(g)
    g
    cpds = [
            LinearGaussianCPD(:X, 0.0, 0.1),
            LinearGaussianCPD(:Y, [:X], [1.0], 0.0, 0.1)
           ]
    bn = BayesNet(cpds; use_topo_sort=false)
    normal_samples = rand(bn, 1000)
    bn, normal_samples
end

bn0, df = get_data()
x, y = eachcol(df)
X = x'
@≥ X, y Array{Float32}.()
# scatter(x, y, xlims=(-1,1), ylims=(-1,1))

#-- fit model

bn = deepcopy(bn0)

