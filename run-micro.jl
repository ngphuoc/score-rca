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
            cpd = RootCPD(nodes[j], Gamma(2, 2))
            push!(dag, cpd)
        else  # inner node/leaf
            ii = findall(A[:, j])
            parents = nodes[ii]
            a = randn(length(parents))
            # cpd = LinearCPD(nodes[j], parents, a, Gamma(2, 2))
            pa_size, hidden = length(parents), 100
            mlp = Chain(
                        Dense(pa_size => hidden, activation, bias=false),
                        Dense(hidden => 1, bias=false),
                       ) |> f64
            cpd = MlpCPD(nodes[j], parents, mlp, Gamma(2, 2))
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

function micro_service_data_v1(; args)
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

"""
TODO? return data, noise, and ∇noise
"""
function draw_normal_perturbed_anomaly(g, n_anomaly_nodes; args)
    d = length(g.cpds)
    #-- normal data
    ε = sample_noise(g, args.n_samples)
    x = forward(g, ε)

    #-- perturbed data, 3σ
    g3 = deepcopy(g)
    for cpd = g3.cpds
        if isempty(parents(cpd))  # perturb root nodes
            cpd.d = scale3(cpd.d)
        end
    end
    ε3 = sample_noise(g3, args.n_samples)
    x3 = forward(g3, ε3)

    #-- select anomaly nodes
    ga = deepcopy(g)
    anomaly_nodes = sample(1:d, n_anomaly_nodes, replace=false)
    a = anomaly_nodes |> first
    for a in anomaly_nodes
        # ga.cpds[a].d = Uniform(3, 5)
        cpd = ga.cpds[a]
        cpd.d = scale3(cpd.d)
    end

    #-- anomaly data
    εa = sample_noise(ga, 20args.n_anomaly_samples)
    xa = forward(ga, εa)
    εa
    ds = [cpd.d for cpd in g.cpds]
    za = @> zval.(ds, eachrow(εa)) hcats transpose
    anomaly_nodes
    z2 = za[anomaly_nodes, :]
    @> abs.(z2) .> 3 minimum(dims=1) vec
    ia = @> abs.(z2) .> 3 minimum(dims=1) vec findall
    ia = ia[1:args.n_anomaly_samples]
    εa = εa[:, ia]
    xa = xa[:, ia]
    y = x - ε
    y3 = x3 - ε3
    ya = xa - εa
    return ε, x, y, ε3, x3, y3, εa, xa, ya, anomaly_nodes
end

n_anomaly_nodes = 1
X = x
μx, σx = @> X mean(dims=2), std(dims=2);
normalise_x(x) = @. (x - μx) / σx
scale_ε(ε) = @. ε / σx
@≥ X, x, xa normalise_x.();
@≥ ε, εa scale_ε.();

fname = "results/micro-service-v5.csv"

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-siren.jl")

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-bigen.jl")

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-causalrca.jl")

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-circa.jl")

g, nodes, x, ε, xa, εa, anomaly_nodes = micro_service_data(; args);
include("method-traversal.jl")

