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

if !isempty(strip(args.load_path))
    @info "Loading data and models from path" args.load_path args.pkl_path
    #-- py data
    dic = pickle_load(args.pkl_path)
    ground_truth_dag, target_node, training_data, ordered_nodes, topo_path = dic
    #-- jl data
    BSON.@load args.load_path args sub_nodes all_nodes target g0 v0 train_df mlp unet

else
    @info "Creating new data and training new models"
    #-- py data
    n_nodes = round(Int, min_depth^2 / 3 + 1)
    ground_truth_dag, target_node, training_data, ordered_nodes, topo_path = get_data(min_depth, n_nodes)
    logistic
    fcm = @> get_fcms(ground_truth_dag) only

    #-- jl data
    sub_nodes = Symbol.(collect(topo_path))
    all_nodes = Symbol.(collect(ordered_nodes))
    target = Symbol(target_node)
    g0, v0 = from_pydigraph(ground_truth_dag, ordered_nodes)
    i0(node) = i2(node, v0)
    train_df = df_(training_data)

    #-- train models
    B = adjacency_matrix(g0)
    @info "Creating unet and mlp models"
    # mlp = GroupMlpRegression(B)
    mlp = GroupMlpRegression(B, fcm)
    unet = GroupMlpUnet(B)
    unet = train_score_model_from_ground_truth_dag(mlp, unet, train_df; args)

    #-- save
    if !isempty(strip(args.save_path))
        @info "Saving data and models to path" args.save_path args.pkl_path
        pickle_save(args.pkl_path, (; ground_truth_dag, target_node, training_data, ordered_nodes, topo_path))
        @≥ mlp, unet cpu.();
        BSON.@save args.save_path args sub_nodes all_nodes target g0 v0 train_df mlp unet
    end
end

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

    #-- normal data
    x, y = eachcol(train_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

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


##-- eval mlp
#X = @> train_df Array;
#loader = DataLoader((X',); args.batchsize, shuffle=true)
#eval_mlp(mlp, loader)

#normal_samples = rand(bn0, min_depth*500)
#learned_dag = fit_dag!(pydeepcopy(ground_truth_dag), pytable(normal_samples))
#df_normal_samples = normal_samples[!, sub_nodes]
#pred_method, topo_path = get_noise_dependent_function(learned_dag, target_node)
#n_outliers = ceil(Int, 0.1d)
#n = e = 0
#n_outlier_nodes = n_outliers
#anomaly_nodes = select_outlier_node(bn0, sub_nodes; n_outlier_nodes)

##--  RUN EXP

#n_nodes0 = length(bn0.cpds)
#n_samples0 = 100 * n_nodes0  # n observations
#n_outliers = n_outlier_nodes
#max_k = n_outlier_nodes
#overall_max_k = max_k + 1
#bn = create_model_from_sub_graph(g0, sub_nodes, all_nodes, df_normal_samples; args)

#n_nodes = length(bn.cpds)
#n_samples = 100 * n_nodes  # n observations

#outlier_bn = deepcopy(bn0)
#assign_outlier_noise!(outlier_bn, anomaly_nodes)
#outlier_samples = rand(outlier_bn, n_outlier_samples)
#df_outlier_samples = outlier_samples[!, sub_nodes]

##--  Get noise dependent functions, with batch size n

#ξs = zeros_ξs(bn)
#ii = [getindex.([bn.name_to_index], parents(cpd)) for cpd in bn.cpds]
#d = length(bn.cpds)
#ws = [get_w(cpd, ξ) for (cpd, ξ) in zip(bn.cpds, ξs)]
#X = Array(df_normal_samples)
#scorer = fit(ZOutlierScore, X[:, end])
#Z = Array(df_outlier_samples)
#leaf_marginal_scores = @> scorer(X[:, end]) mean, std round.(digits=2)
#outlier_marginal_scores = @> scorer(Z[:, end]) mean, std round.(digits=2)
#μ0, σ0 = leaf_marginal_scores
#μ1, σ1 = outlier_marginal_scores

#function ϵ_func(ϵ)
#    X = zeros(size(ϵ, 1), 0)
#    for j = 1:d
#        cpd = bn.cpds[j]
#        x = X[:, ii[j]]  # use ii to avoid Zygote mutating error
#        w = ws[j]
#        y = x * w + ϵ[:, j]
#        X = hcat(X, y)
#    end
#    return scorer(X[:, end]) |> mean
#end

#function ξ_func(ξs, ϵ)
#    X = zeros(size(ϵ, 1), 0)
#    for j = 1:d
#        ξ = ξs[j]
#        cpd = bn.cpds[j]
#        x = X[:, ii[j]]
#        w = get_w(cpd, ξ)
#        y = x * w + ϵ[:, j]
#        X = hcat(X, y)
#    end
#    return scorer(X[:, end]) |> mean
#end

#Xo = Array(df_outlier_samples)

##--  STEP 1: ξ score

#function i_(node)
#    indexin([node], PyList(topo_path))[1]
#end

#ϵ = infer_ϵ(bn, Xo)
#ϵ0 = zero(ϵ)
## ξ0 = zeros_ξs(bn)
#ξ0 = @> infer_ξ(bn, df_normal_samples) unpack_ξ
#ξ = infer_ξ(bn, df_outlier_samples)
#ξs = unpack_ξ(ξ)
## init reference
#ξ′s = []
#t = 0.5
#for t = 0.0:1.0
#    ξt = ξ0 + t .* (ξs - ξ0)
#    ξ_func(ξt, ϵ)
#    ξ′, = Zygote.gradient(ξ_func, ξt, ϵ)
#    push!(ξ′s, nothing2empties(ξ′))
#end
#ξs_score = @> +(ξ′s...) / length(ξ′s)
#ξ_score = pack_ξ(ξs_score)

## get_ground_truth_ranking for edges
#function get_ξ_rankings()
#    ws = get_noise_coefficient(ground_truth_dag, target_node) |> PyDict
#    ϵ = infer_ϵ(bn, Xo)
#    ξ = infer_ξ(bn, df_outlier_samples)
#    ξs = unpack_ξ(ξ)
#    tmp = Dict(Symbol(node) => float(w) .* ξs[i_(node)] for (node, w) in ws)
#    tmp = @> getindex.([tmp], sub_nodes) vcats
#    ranking = sortperm(tmp, rev=true)
#    score = zeros(length(ranking))
#    for q in 1:max_k
#        score[ranking[q]] = overall_max_k - q
#    end
#    return score
#end
#gt_ξ_score = get_ξ_rankings()

#grad_edge = ndcg_score(gt_ξ_score', ((ξ .- ξ0) .* ξ_score)', k=n_outliers)



##--  STEP 2: ϵ score

#ξs = ξs
#ϵ = @> infer_ϵ(bn, Xo; ξs)
#@≥ ϵ reshape(1, size(ϵ)...) permutedims([2, 1, 3])
#ir = rand(1:n_samples0, n_reference_samples)
#ϵ0 = infer_ϵ(bn, X[ir, :]; ξs)
#@≥ ϵ0 reshape(1, size(ϵ0)...)
#ϵ′s = []
#for t = 0.0:1.0
#    ϵt = ϵ0 .+ t * (ϵ .- ϵ0)
#    @≥ ϵt reshape(:, d)
#    ϵ_func(ϵt)
#    ϵ′, = Zygote.gradient(ϵ_func, ϵt)
#    push!(ϵ′s, ϵ′)
#end
#ϵ_scores = @> average(ϵ′s...) reshape(n_outlier_samples, n_reference_samples, :) mean(dims=2) mat

## get_ground_truth_ranking version for both nodes and edges
#function get_ϵ_rankings()
#    ws = get_noise_coefficient(ground_truth_dag, target_node) |> PyDict
#    ϵ = infer_ϵ(bn, Xo)
#    j = 1
#    scores = Vector{Float64}[]
#    for j = 1:size(ϵ, 1)
#        tmp = Dict(node => w * ϵ[j, i_(node)] for (node, w) in ws)
#        ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]
#        score = zeros(length(sub_nodes))
#        for q in 1:max_k
#            iq = findfirst(==(Symbol(ranking[q])), sub_nodes)
#            score[iq] = overall_max_k - q
#        end
#        push!(scores, score)
#    end
#    return scores
#end

#gt_ϵ_scores = @> get_ϵ_rankings() transpose.() vcats
#ϵ = @> infer_ϵ(bn, Xo; ξs)
#grad_node = ndcg_score(gt_ϵ_scores, (ϵ .- ϵ0) .* ϵ_scores, k=n_outliers)

#scorerpy = ZOutlierScorePy(X[:, end])
#@> scorerpy.score(X[:, end]) collect float.() mean, std round.(digits=2)
#@> scorerpy.score(Z[:, end]) collect float.() mean, std round.(digits=2)
#ref_samples = DataFrame(mat(ϵ0, dims=2), sub_nodes)

##--  shapley
#contributions = our_approach_rankings(learned_dag, target_node, pytable(outlier_samples), target_prediction_method=pred_method, nodes_order=topo_path, zscorer=scorerpy, ref_samples=pytable(ref_samples), approximation_method=ShapleyApproximationMethods.AUTO)
#shapley_node_scores = get_node_scores(contributions)

##--   sampling
#contributions = our_approach_rankings(learned_dag, target_node, pytable(outlier_samples), target_prediction_method=pred_method, nodes_order=topo_path, zscorer=scorerpy, ref_samples=pytable(ref_samples), approximation_method=ShapleyApproximationMethods.SUBSET_SAMPLING)
#sampling_node_scores = get_node_scores(contributions)

#############  permutation
#contributions = our_approach_rankings(learned_dag, target_node, pytable(outlier_samples), target_prediction_method=pred_method, nodes_order=topo_path, zscorer=scorerpy, ref_samples=pytable(ref_samples), approximation_method=ShapleyApproximationMethods.PERMUTATION)
#permutation_node_scores = get_node_scores(contributions)

##--  Naive RCA
#contributions = naive_approach(learned_dag, topo_path, pytable(outlier_samples))
#naive_node_scores = get_node_scores(contributions)

#shapley_node = ndcg_score(gt_ϵ_scores, shapley_node_scores, k=n_outliers)
#sampling_node = ndcg_score(gt_ϵ_scores, sampling_node_scores, k=n_outliers)
#permutation_node = ndcg_score(gt_ϵ_scores, permutation_node_scores, k=n_outliers)
#naive_node = ndcg_score(gt_ϵ_scores, naive_node_scores, k=n_outliers)

#@≥ shapley_node, sampling_node, permutation_node, naive_node, grad_node float.() round.(digits=3)


##--  STEP 3: compute scores

#shapley_w_score = shapley_node_scores'shapley_node_scores
#shapley_edge_score = shapley_w_score[edges_ci]
#shapley_edge = ndcg_score(gt_ξ_score', shapley_edge_score', k=n_outliers)

#sampling_w_score = sampling_node_scores'sampling_node_scores
#sampling_edge_score = sampling_w_score[edges_ci]
#sampling_edge = ndcg_score(gt_ξ_score', sampling_edge_score', k=n_outliers)

#permutation_w_score = permutation_node_scores'permutation_node_scores
#permutation_edge_score = permutation_w_score[edges_ci]
#permutation_edge = ndcg_score(gt_ξ_score', permutation_edge_score', k=n_outliers)

#naive_w_score = naive_node_scores'naive_node_scores
#naive_edge_score = naive_w_score[edges_ci]
#naive_edge = ndcg_score(gt_ξ_score', naive_edge_score', k=n_outliers)

#@≥ grad_edge, shapley_edge, sampling_edge, permutation_edge, naive_node, naive_edge float.() round.(digits=3)

#df = DataFrame(n_nodes0=Int[], n_nodes=Int[], n=Int[], e = Int[],
#               shapley_node = Float64[], shapley_edge = Float64[],
#               sampling_node = Float64[], sampling_edge = Float64[],
#               permutation_node = Float64[], permutation_edge = Float64[],
#               naive_node = Float64[], naive_edge = Float64[],
#               grad_node = Float64[], grad_edge = Float64[],
#               k = Int[], μ0 = Float64[], μ1 = Float64[])

#k = 1
#for k=1:min_depth
#    local shapley_node = ndcg_score(gt_ϵ_scores, shapley_node_scores, k=k)
#    local sampling_node = ndcg_score(gt_ϵ_scores, sampling_node_scores, k=k)
#    local permutation_node = ndcg_score(gt_ϵ_scores, permutation_node_scores, k=k)
#    local naive_node = ndcg_score(gt_ϵ_scores, naive_node_scores, k=k)
#    local grad_node = ndcg_score(gt_ϵ_scores, ϵ .* ϵ_scores, k=k)
#    local shapley_edge = ndcg_score(gt_ξ_score', shapley_edge_score', k=k)
#    local sampling_edge = ndcg_score(gt_ξ_score', sampling_edge_score', k=k)
#    local permutation_edge = ndcg_score(gt_ξ_score', permutation_edge_score', k=k)
#    local naive_edge = ndcg_score(gt_ξ_score', naive_edge_score', k=k)
#    local grad_edge = ndcg_score(gt_ξ_score', (ξ .* ξ_score)', k=k)
#    push!(df, @> [n_nodes0, n_nodes, n_outlier_nodes, n_outlier_edges,
#                  shapley_node, shapley_edge,
#                  sampling_node, sampling_edge,
#                  permutation_node, permutation_edge,
#                  naive_node, naive_edge,
#                  grad_node, grad_edge,
#                  k, μ0, μ1] float.() round.(digits=3))
#end

#println(df);

#fname = "results/random-graphs.csv"
#CSV.write(fname, df, header=!isfile(fname), append=true)

