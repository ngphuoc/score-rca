using Flux: Data
using Graphs, BayesNets, Flux, PythonCall
using Parameters: @unpack
@unpack truncexpon, halfnorm = pyimport("scipy.stats")
@unpack random = pyimport("numpy")
include("lib/utils.jl")
include("bayesnets-extra.jl")
include("bayesnets-fit.jl")
include("data-rca.jl")

#-- 1. dag, linear model, and noises

function micro_service_dag()
    #-- a. Create digraph
    nodes = [
             "API",
             "Auth Service",
             "Caching Service",
             "Customer DB",
             "Order DB",
             "Order Service",
             "Product DB",
             "Product Service",
             "Shipping Cost Service",
             "Website",
             "www",
            ]
    edges = [
             ("www", "Website"),
             ("Auth Service", "www"),
             ("API", "www"),
             ("Customer DB", "Auth Service"),
             ("Customer DB", "API"),
             ("Product Service", "API"),
             ("Auth Service", "API"),
             ("Order Service", "API"),
             ("Shipping Cost Service", "Product Service"),
             ("Caching Service", "Product Service"),
             ("Product DB", "Caching Service"),
             ("Customer DB", "Product Service"),
             ("Order DB", "Order Service")]
    nodes = fmap(Symbol, nodes)
    edges = fmap(Symbol, edges)
    d = length(nodes)
    nodeidx = Dict(nodes[i] => i for i=1:length(nodes))
    g = DiGraph(d)
    for (b, e) = edges
        add_edge!(g, nodeidx[b], nodeidx[e]);
    end
    g

    #-- topological_sort
    sorted = topological_sort(g)
    A = adjacency_matrix(g)
    A = A[sorted, sorted]
    nodes = nodes[sorted]
    g = DiGraph(A)

    #-- b. Create BayesNet
    dag = BayesNet()
    d = nv(g)
    A = @> adjacency_matrix(g) Matrix{Bool}
    for j = 1:d
        if sum(A[:, j]) == 0  # root
            cpd = RootCPD(nodes[j], Gamma(1, 2))
            push!(dag, cpd)
        else  # inner node/leaf
            ii = findall(A[:, j])
            parents = nodes[ii]
            a = randn(length(parents))
            cpd = LinearCPD(nodes[j], parents, a, Gamma(1, 2))
            push!(dag, cpd)
        end
    end
    return dag, nodes
end

function unobserved_intrinsic_latencies_normal(num_samples)
    d = Dict(
             "Website"               => truncexpon.rvs(size=num_samples, b=3, scale=0.2),
             "www"                   => truncexpon.rvs(size=num_samples, b=2, scale=0.2),
             "API"                   => halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
             "Auth Service"          => halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
             "Product Service"       => halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
             "Order Service"         => halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
             "Shipping Cost Service" => halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
             "Caching Service"       => halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
             "Order DB"              => truncexpon.rvs(size=num_samples, b=5, scale=0.2),
             "Customer DB"           => truncexpon.rvs(size=num_samples, b=6, scale=0.2),
             "Product DB"            => truncexpon.rvs(size=num_samples, b=10, scale=0.2)
            )
    fmap(Array ∘ PyArray, d)
end

# "Caching Service" => 2 + normal delay
function unobserved_intrinsic_latencies_anomalous(num_samples)
    d = Dict(
             "Website"               => truncexpon.rvs(size=num_samples, b=3, scale=0.2),
             "www"                   => truncexpon.rvs(size=num_samples, b=2, scale=0.2),
             "API"                   => halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
             "Auth Service"          => halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
             "Product Service"       => halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
             "Order Service"         => halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
             "Shipping Cost Service" => halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
             "Caching Service"       => 2 + halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
             "Order DB"              => truncexpon.rvs(size=num_samples, b=5, scale=0.2),
             "Customer DB"           => truncexpon.rvs(size=num_samples, b=6, scale=0.2),
             "Product DB"            => truncexpon.rvs(size=num_samples, b=10, scale=0.2)
            )
    fmap(Array ∘ PyArray, d)
end

function create_observed_latency_data(unobserved_intrinsic_latencies, nodes; w_auth_api = 1, w_cus_prod = 1)
    observed_latencies = Dict()
    observed_latencies["Product DB"] = unobserved_intrinsic_latencies["Product DB"]
    observed_latencies["Customer DB"] = unobserved_intrinsic_latencies["Customer DB"]
    observed_latencies["Order DB"] = unobserved_intrinsic_latencies["Order DB"]
    observed_latencies["Shipping Cost Service"] = unobserved_intrinsic_latencies["Shipping Cost Service"]
    observed_latencies["Caching Service"] = rand([0, 1], length(observed_latencies["Product DB"])) .* observed_latencies["Product DB"] .+ unobserved_intrinsic_latencies["Caching Service"]
    observed_latencies["Product Service"] = max.(observed_latencies["Shipping Cost Service"], observed_latencies["Caching Service"], w_cus_prod .* observed_latencies["Customer DB"]) .+ unobserved_intrinsic_latencies["Product Service"]
    observed_latencies["Auth Service"] = observed_latencies["Customer DB"] .+ unobserved_intrinsic_latencies["Auth Service"]
    observed_latencies["Order Service"] = observed_latencies["Order DB"] .+ unobserved_intrinsic_latencies["Order Service"]
    observed_latencies["API"] = observed_latencies["Product Service"] \
                                .+ observed_latencies["Customer DB"] \
                                .+ w_auth_api .* observed_latencies["Auth Service"] \
                                .+ observed_latencies["Order Service"] \
                                .+ unobserved_intrinsic_latencies["API"]
    observed_latencies["www"] = observed_latencies["API"] .+ observed_latencies["Auth Service"] .+ unobserved_intrinsic_latencies["www"]
    observed_latencies["Website"] = observed_latencies["www"] .+ unobserved_intrinsic_latencies["Website"]
    df = DataFrame(observed_latencies)
    df[!, nodes]
end

function micro_service_data(; args)
    bn, nodes = micro_service_dag()
    ε = unobserved_intrinsic_latencies_normal(args.n_samples)
    x = create_observed_latency_data(ε, nodes)
    εa = unobserved_intrinsic_latencies_anomalous(args.n_anomaly_samples)
    xa = create_observed_latency_data(εa, nodes)
    anomaly_nodes = indexin([Symbol("Caching Service")], nodes)
    X = @> Array(x)' Array
    Distributions.fit!(bn, X)
    @assert Symbol.(names(x)) == nodes
    target_node = Symbol("Website")
    ε = DataFrame(ε)[!, nodes]
    εa = DataFrame(εa)[!, nodes]
    @≥ x, ε, xa, εa Array.() transpose.()
    bn, nodes, x, ε, xa, εa, anomaly_nodes
end
cf
fname = "results/micro-service.csv"

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-circa.jl")
CSV.write(fname, df, header=!isfile(fname), append=true)

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-circa.jl")
CSV.write(fname, df, header=!isfile(fname), append=true)

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-circa.jl")
CSV.write(fname, df, header=!isfile(fname), append=true)

