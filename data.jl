include("./lib/utils.jl")
include("./lib/nn.jl")
include("./lib/nnlib.jl")
include("./lib/graph.jl")
include("./dlsm-losses.jl")
using Flux
using Flux: crossentropy
using Flux.Data: DataLoader
using DataFrames, Distributions, BayesNets, CSV, Tables, FileIO, JLD2

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

    # create BayesNets
    cpds = map(1:d) do j
        ii = parent_indices(B, j)
        node = v[j]
        if length(ii) == 0
            # fit(StaticCPD{Normal}, df_(training_data), node)
            cpd = StaticCPD(node, Normal(0, 1))
        else
            pa = v[ii]
            # fit(LinearBayesianCPD, df_(training_data), node, pa)
            # cpd = fit(LinearBayesianCPD, df_(training_data), node, pa)
            cpd = fit(LinearBayesianCPD, df_(training_data), node, pa)
            cpd.ps[2] = 0.01I(length(pa)) |> Array
            cpd.ps[end] = 1.0
            cpd
        end
    end
    bn0 = BayesNet(cpds; use_topo_sort=false)
    return bn0

    # sample data

    bn0 = create_model_from_ground_truth_dag(g0, v, training_data)
    normal_samples = rand(bn0, min_depth*500)

end

function create_model_from_ground_truth_dag(g0, v, training_data)
    d = length(v)
    B = adjacency_matrix(g0)
    g = DiGraph(B)
    cpds = map(1:d) do j
        ii = parent_indices(B, j)
        node = v[j]
        if length(ii) == 0
            # fit(StaticCPD{Normal}, df_(training_data), node)
            cpd = StaticCPD(node, Normal(0, 1))
        else
            pa = v[ii]
            # fit(LinearBayesianCPD, df_(training_data), node, pa)
            cpd = fit(LinearBayesianCPD, df_(training_data), node, pa)
            cpd.ps[2] = 0.01I(length(pa)) |> Array
            cpd.ps[end] = 1.0
            cpd
        end
    end
    bn0 = BayesNet(cpds; use_topo_sort=false)
    return bn0
end

function fit_dag!(bn::BayesNet{CPD}, normal_samples)
    cpds = map(bn.cpds) do cpd
        @show typeof(cpd)
        node = name(cpd)
        pa = parents(cpd)
        if length(pa) == 0
            fit(StaticCPD{Normal}, normal_samples, node)
        else
            fit(LinearBayesianCPD, normal_samples, node, pa)
        end
    end
    bn0 = BayesNet(cpds; use_topo_sort=false)
    return bn0
end


function get_dataset()
    X, y = nmoons(Float64, n_data, 2, ε=0.25, d=2, repulse=(-0.25,0.0))
    DataLoader((X, y), batchsize=args.batchsize, shuffle=true)
end

ds = get_dataset()
(batch, labels) = ds |> first |> gpu
classifier_model = NCClassifier(; args) |> to_device;
score_model = NCScore(; args) |> to_device;



