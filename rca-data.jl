using StatsBase

function sample_natural_number(; init_mass)
    current_mass = init_mass
    probability = rand()
    k = 1
    is_searching = true
    while is_searching
        if probability <= current_mass
            return k
        else
            k += 1
            current_mass += 1 / (k ^ 2)
        end
    end
end

"""
n_root_nodes = 2
n_downstream_nodes = 4
"""
function random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, scale, hidden, activation=Flux.σ)
    @info "random_nonlinear_dag_generator"
    dag = BayesNet()
    for i in 1:n_root_nodes
        cpd = StaticCPD(Symbol("X$i"), Normal(0, 1))
        push!(dag, cpd)
    end
    for i in 1:n_downstream_nodes
        parents = sample(names(dag), min(sample_natural_number(init_mass=0.6), length(dag)), replace=false)
        # Random mechanism
        pa_size = length(parents)
        W1 = rand(Uniform(0.5, 2.0), (hidden, pa_size))
        W1[rand(size(W1)...) .< 0.5] .*= -1
        W2 = rand(Uniform(0.5, 2.0), (1, hidden))
        W2[rand(size(W2)...) .< 0.5] .*= -1
        mlp = Chain(
                    Dense(pa_size => hidden, activation, bias=false),
                    Dense(hidden => 1, bias=false),
                   )
        mlp[1].weight .= W1
        mlp[2].weight .= W2
        cpd = MlpGaussianCPD(Symbol("X$(i + n_root_nodes)"), parents, mlp, scale)
        push!(dag, cpd)
    end
    return dag
end

function Base.getindex(bn::BayesNet, node::Symbol)
    bn.cpds[bn.name_to_index[node]]
end

function draw_normal_anomaly(dag; args)
    normal_df = rand(dag, args.n_samples)
    #-- select anomaly nodes
    anomaly_dag = deepcopy(dag)
    anomaly_nodes = sample(names(dag), args.n_anomaly_nodes)
    for anomaly_node = anomaly_nodes
        anomaly_dag[anomaly_node]
    end


    anomaly_df = rand(anomaly_dag, args.n_samples)

    return normal_df, anomaly_df

    causal_graph, n_samples, k, list_of_potential_anomaly_nodes
    drawn_samples = Dict()
    drawn_noise_samples = Dict()
    lambdas = Dict()

    noises = numpy.random.uniform(3, 5, n_samples)
    anomaly_nodes = random.sample(list_of_potential_anomaly_nodes, k)

    for (i, node) in enumerate(networkx.topological_sort(causal_graph.graph))
        causal_model = causal_graph.causal_mechanism(node)

        if is_root_node(causal_graph.graph, node)
            if node in anomaly_nodes
                drawn_noise_samples[node] = numpy.array(noises)
            else
                drawn_noise_samples[node] = numpy.zeros(n_samples)
            end

            drawn_samples[node] = drawn_noise_samples[node]
        else
            if node in anomaly_nodes
                drawn_noise_samples[node] = numpy.array(noises)
            else
                drawn_noise_samples[node] = numpy.zeros(n_samples)
            end

            parent_samples = column_stack_selected_numpy_arrays(drawn_samples, get_ordered_predecessors(causal_graph.graph, node))

            drawn_samples[node] = causal_model.evaluate(parent_samples, drawn_noise_samples[node])
        end
    end
end

