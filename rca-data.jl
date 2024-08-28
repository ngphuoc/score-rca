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
        cpd = RootCPD(Symbol("X$i"), Normal(0, 1))
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
        cpd = MlpCPD(Symbol("X$(i + n_root_nodes)"), parents, mlp, Normal(0f0, Float32(scale)))
        push!(dag, cpd)
    end
    return dag
end

function Base.getindex(bn::BayesNet, node::Symbol)
    bn.cpds[bn.name_to_index[node]]
end

function draw_normal_perturbed_anomaly(dag; args)
    #-- normal data
    normal_df = rand(dag, args.n_samples)

    #-- perturbed data, 3σ
    g = deepcopy(dag)
    for cpd = g.cpds
        if isempty(parents(cpd))  # perturb root nodes
            cpd.d = Normal(cpd.d.μ, args.perturbed_scale * cpd.d.σ)
        end
    end
    perturbed_df = rand(g, args.n_samples)

    #-- select anomaly nodes
    g = deepcopy(dag)
    anomaly_nodes = sample(names(g), args.n_anomaly_nodes)
    for a = anomaly_nodes
        g[a].d = Uniform(3, 5)
    end
    #-- anomaly data
    anomaly_df = rand(g, args.n_anomaly_samples)

    return normal_df, perturbed_df, anomaly_df  # drawn_noise_samples
end

using Distributions, Statistics, Plots, LaTeXStrings, Plots.PlotMeasures, ColorSchemes
gr()

function plot_3data(train_df, perturbed_df, anomaly_df; xlim, ylim)
    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

    #-- plot data
    x, y = eachcol(train_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"data $(x, y)$")
    x, y = eachcol(perturbed_df)
    pl_perturbed = scatter(x, y; xlab=L"x", ylab=L"y", title=L"perturbed data $(x, y)$")
    x, y = eachcol(anomaly_df)
    pl_anomaly = scatter(x, y; xlab=L"x", ylab=L"y", title=L"anomaly data $(x, y)$")
    @> Plots.plot(pl_data, pl_perturbed, pl_anomaly; xlim, ylim, size=(1000, 800)) savefig("fig/3data-2d.png")
end

