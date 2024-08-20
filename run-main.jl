include("./imports.jl")
include("./rca.jl")
include("./lib/diffusion.jl")
include("./random-graph-datasets.jl")

datetime_prefix = @> string(now())[1:16] replace(":"=>"h")

args = @env begin
    # Denoising
    input_dim = 2
    output_dim = 2
    hidden_dim = 32  # hiddensize factor
    n_layers = 3
    embed_dim = 32  # hiddensize factor
    scale=30.0f0  # RandomFourierFeatures scale
    σ_max = 5.0
    σ_min = 1e-3
    lr_mlp = 1e-1  # learning rate
    lr_unet = 1e-1  # learning rate
    decay = 1e-5  # weight decay parameter for AdamW
    to_device = Flux.gpu
    batchsize = 32
    epochs = 300
    save_path = "data/exp2d.bson"
    load_path = ""
    pkl_path = "data/exp2d.pkl"
    # RCA
    min_depth = 2  # minimum depth of ancestors for the target node
    n_root_nodes = 1  # num_root_nodes
    n_samples = 500  # n observations
    n_outlier_samples = 8  # n faulty observations
    has_node_outliers = true  # node outlier setting
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    noise_scale = 1.0
    seed = 1  #  random seed
end

#--  CREATE DATASETS

#if !isempty(strip(args.load_path))
#    @info "Loading data and models from path" args.load_path args.pkl_path
#    #-- py data
#    dic = pickle_load(args.pkl_path)
#    ground_truth_dag, target_node, training_data, ordered_nodes, topo_path = dic
#    #-- jl data
#    BSON.@load args.load_path args sub_nodes all_nodes target g0 v0 train_df mlp unet

#else
    @info "Creating new data and training new models"
    #-- py data
    n_nodes = round(Int, min_depth^2 / 3 + 1)
    scale, hidden = 0.001, 2
    ground_truth_dag, target_node, training_data, ordered_nodes, topo_path = get_data(min_depth, n_nodes; scale, hidden)
    fcm = @> get_fcms(ground_truth_dag) only

    #-- jl data
    sub_nodes = Symbol.(collect(topo_path))
    all_nodes = Symbol.(collect(ordered_nodes))
    target = Symbol(target_node)
    g0, v0 = from_pydigraph(ground_truth_dag, ordered_nodes)
    i0(node) = i2(node, v0)
    train_df = df_(training_data)
    X = @> train_df Array transpose Array;
    μX, σX = @> X mean(dims=2), std(dims=2);
    X = @. (X - μX) / σX  # normalise

    #-- train models
    hidden_dims = [2, ]
    paj_mask = B = @> adjacency_matrix(g0) Matrix{Bool}
    @info "Creating unet and mlp models"
    mlp = GroupMlpRegression(B; μX, σX, hidden_dims, activation=Flux.tanh)
    # no fcm for node 1
    mlp.mlp[1].weight[:, :, 1] .= 0
    mlp.mlp[2].weight[:, :, 1] .= 0
    # fcm for node 2
    # with input node 1
    mlp.mlp[1].weight[:, 1, 2] .= fcm[1].weight
    mlp.mlp[1].weight[:, 2, 2] .= 0
    mlp.mlp[1].bias .= 0
    # fcm layer 2
    mlp.mlp[2].weight[:, :, 2] .= fcm[2].weight
    mlp.mlp[2].bias .= 0
    # mlp = GroupMlpRegression(Matrix{Bool}(B), fcm)
    unet = GroupMlpUnet(B)

    mlp, unet = train_score_model_from_ground_truth_dag(mlp, unet, train_df; args)

    ##-- save
    #if !isempty(strip(args.save_path))
    #    @info "Saving data and models to path" args.save_path args.pkl_path
    #    pickle_save(args.pkl_path, (; ground_truth_dag, target_node, training_data, ordered_nodes, topo_path))
    #    @≥ mlp, unet cpu.();
    #    BSON.@save args.save_path args sub_nodes all_nodes target g0 v0 train_df mlp unet
    #end
#end

using Distributions, Statistics, StatsPlots, Plots, LaTeXStrings, Plots.PlotMeasures, ColorSchemes
gr()

function arrow0!(x, y, u, v; as=0.07, lw=1, lc=:black, la=1)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = as*nuv*v4, as*nuv*v5
    plot!([x,x+u], [y,y+v], lw=lw, lc=lc, la=la)
    plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], lw=lw, lc=lc, la=la)
    plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], lw=lw, lc=lc, la=la)
end

function plot_data_gradients(train_df, mlp, unet; xlims, ylims)
    @≥ mlp, unet gpu.();

    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlims, ylims, color=:seaborn_deep, markersize=2, leg=nothing)

    #-- plot data
    x, y = eachcol(train_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

    #-- plot r2
    x = @> train_df Array transpose Array gpu;
    d = size(x, 1)
    total_loss = 0.0
    xj = unsqueeze(x, 1);
    @≥ x unsqueeze(2) repeat(1, d, 1)
    x̂ = mlp(x)
    y, ŷ = @> xj, x̂ getindex.(1,2,:) cpu.() vec.();
    pl_r2 =  scatter(y, ŷ; xlab=L"y", ylab=L"\hat{y}", title=L"Regression $R^2=%$(round(float(r2_score(y, ŷ)), digits=2))$")

    #-- plot perturbations
    x = @> train_df Array transpose Array gpu;
    d = size(x, 1)
    X, batchsize = size(x, 1), size(x)[end]
    j_mask = @> I(X) Matrix{Float32} gpu;
    xj = x[[2], :]
    t = rand!(similar(xj)) .* (1f0 - 1f-5) .+ 1f-5
    σ_t = marginal_prob_std(t)
    z = 2rand!(similar(x)) .- 1;
    x̃ = x .+ σ_t .* z
    x, y = eachrow(cpu(x̃))
    pl_perturbed_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Perturbed data $(x, y)$")

    #-- plot gradients
    x = @> Iterators.product(range(xlims..., length=30), range(ylims..., length=30)) collect vec;
    x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
    d = size(x, 1)
    xj = unsqueeze(x, 1);
    @≥ x unsqueeze(2) repeat(1, d, 1)
    t = fill!(similar(xj), 0.1)
    σ_t = marginal_prob_std(t)
    J = @> unet(x, t)[:, 2, :]
    x = @> squeeze(xj, 1)
    @≥ J, x cpu.()
    x, y = eachrow(x);
    u, v = eachrow(0.05J);
    _, _, _, s = @> norm2(J, dims=1) maximum, minimum, mean, std

    pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
    arrow0!.(x, y, u, v; as=0.3, lw=1.0);
    @> Plots.plot(pl_data, pl_perturbed_data, pl_r2, pl_gradient; xlims, ylims, size=(1000, 800)) savefig("fig/$datetime_prefix-2d.png")
end

xlims = ylims = (-15, 15)
plot_data_gradients(train_df, mlp, unet; xlims, ylims)

