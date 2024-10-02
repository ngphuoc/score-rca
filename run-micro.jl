using Graphs, BayesNets, Flux, PythonCall
include("lib/utils.jl")
include("bayesnets-extra.jl")
include("bayesnets-fit.jl")
using Parameters: @unpack
@unpack truncexpon, halfnorm = pyimport("scipy.stats")
@unpack random = pyimport("numpy")

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
    observed_latencies["Caching Service"] = PyArray(random.choice([0, 1], size=(length(observed_latencies["Product DB"]),), p=[.5, .5])) .* observed_latencies["Product DB"] .+ unobserved_intrinsic_latencies["Caching Service"]
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

function micro_service_data(n_samples)
    bn, nodes = micro_service_dag()
    normal_noise = unobserved_intrinsic_latencies_normal(n_samples)
    normal_data = create_observed_latency_data(normal_noise, nodes)
    names(normal_data)
    X = @> Array(normal_data)' Array
    Distributions.fit!(bn, X)
    @assert Symbol.(names(normal_data)) == nodes
    target_node = Symbol("Website")
    return (bn, target_node, nodes, normal_data, normal_noise)
end

unobserved_intrinsic_latencies = unobserved_intrinsic_latencies_normal(10000)
normal_data = create_observed_latency_data(unobserved_intrinsic_latencies_normal(10000))
outlier_data = create_observed_latency_data(unobserved_intrinsic_latencies_anomalous(1000))

#-- 2. non-linear FCMs, and 2 more scenarios from micro hrelc

#-- 3. runs

