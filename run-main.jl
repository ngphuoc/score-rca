using Revise
include("./imports.jl")
include("./rca.jl")
include("./rca-data.jl")
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
    perturbed_scale = 1f0
    σ_max = 5.0
    σ_min = 1e-3
    lr_regressor = 1e-3  # learning rate
    lr_unet = 1e-3  # learning rate
    decay = 1e-5  # weight decay parameter for AdamW
    to_device = Flux.gpu
    batchsize = 64
    epochs = 100
    save_path = ""
    load_path = "data/exp2d.bson"
    pkl_path = "data/exp2d.pkl"
    # RCA
    min_depth = 2  # minimum depth of ancestors for the target node
    n_root_nodes = 1  # n_root_nodes
    n_samples = 1000  # n observations
    n_anomaly_nodes = 2
    n_anomaly_samples = 100  # n faulty observations
    has_node_outliers = true  # node outlier setting
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    noise_scale = 1.0
    seed = 1  #  random seed
end

if !isempty(strip(args.load_path))
    @info "Loading data and models from path" args.load_path
    BSON.@load args.load_path args dag normal_df perturbed_df anomaly_df μX σX oracle regressor unet

else
    @info "Creating new data and training new models"

    n_nodes = round(Int, min_depth^2 / 3 + 1)
    n_root_nodes = 1
    n_downstream_nodes = n_nodes - n_root_nodes
    scale, hidden = 0.5, 100
    dag = random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, scale, hidden)
    normal_df, perturbed_df, anomaly_df = draw_normal_perturbed_anomaly(dag; args)

    #-- normalise data
    X = @> vcat(normal_df, perturbed_df) Array transpose Array;
    μX, σX = @> X mean(dims=2), std(dims=2);
    normal_df = @. (normal_df - μX') / σX'
    perturbed_df = @. (perturbed_df - μX') / σX'
    anomaly_df = @. (anomaly_df - μX') / σX'

    @> normal_df Array minimum(dims=1), maximum(dims=1)
    @> perturbed_df Array minimum(dims=1), maximum(dims=1)

    # xlim = ylim = (-15, 15)
    # plot_3data(normal_df, perturbed_df, anomaly_df; xlim, ylim)

    #-- train models
    # X = @> vcat(normal_df, perturbed_df) Array transpose Array;
    hidden_dims = [50, ]
    @info "Creating unet and mlp models"
    oracle = GroupOracleRegression(dag)
    regressor = GroupMlpRegression(dag; hidden_dims, activation=Flux.swish)
    df = vcat(normal_df, perturbed_df)
    regressor = train_regressor(regressor, df; args)

    unet = GroupMlpUnet(dag)
    regressor, unet = train_unet(regressor, unet, normal_df; args)

    #-- save
    if !isempty(strip(args.save_path))
        @info "Saving data and models to path" args.save_path
        @≥ oracle, regressor, unet cpu.();
        BSON.@save args.save_path args dag normal_df perturbed_df anomaly_df μX σX oracle regressor unet
    end
end


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


function plot_data_gradients(normal_df, perturbed_df, regressor, oracle, unet, σX, μX; xlim, ylim)
    @≥ regressor, oracle, unet gpu.();

    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

    #-- plot data
    x, y = eachcol(normal_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

    #-- plot r2
    x = @> vcat(normal_df, perturbed_df) Array transpose Array gpu;
    @> x[2, :] minimum, maximum, mean, std
    # x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    total_loss = 0.0
    xj = unsqueeze(x, 1)
    @≥ x unsqueeze(2) repeat(1, d, 1)
    x̂ = regressor(x)
    @assert size(xj) == size(x̂)
    y, ŷ = @> xj, x̂ getindex.(1,2,:) cpu.() vec.();
    pl_r2 =  scatter(y, ŷ; xlab=L"y", ylab=L"\hat{y}", title=L"Regression $R^2=%$(round(float(r2_score(y, ŷ)), digits=2))$")

    #-- plot oracle r2, need to save μX, σX for oracle regressor
    # x = @> vcat(normal_df, perturbed_df) Array transpose Array gpu;
    x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    total_loss = 0.0
    xj = unsqueeze(x, 1);
    @≥ x unsqueeze(2) repeat(1, d, 1)
    @≥ σX, μX gpu.()
    x0 = @. x * σX + μX
    x̂ = (oracle(x0) .- μX') ./ σX'
    @assert size(xj) == size(x̂)
    y, ŷ = @> xj, x̂ getindex.(1,2,:) cpu.() vec.();
    # m = dag.cpds[2].mlp
    # ŷ = @> m(cpu(x[[1], :])) vec
    # y = @> x[2, :] cpu
    pl_oracle =  scatter(y, ŷ; xlab=L"y", ylab=L"\hat{y}", title=L"Oracle Regression $R^2=%$(round(float(r2_score(y, ŷ)), digits=2))$")

    #-- plot perturbations
    x, y = eachcol(perturbed_df)
    pl_perturbed_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Perturbed $3\sigma$ $(x, y)$")

    x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    X, batchsize = size(x, 1), size(x)[end]
    j_mask = @> I(X) Matrix{Float32} gpu;
    xj = x[[2], :]
    t = rand!(similar(xj)) .* (1f0 - 1f-5) .+ 1f-5
    σ_t = marginal_prob_std(t)
    z = 2rand!(similar(x)) .- 1;
    x̃ = x .+ σ_t .* z
    x, y = eachrow(cpu(x̃))
    pl_sm_data = scatter(x, y; xlab=L"x", ylab=L"y", title="Perturbed score matching")

    #-- plot gradients
    x = @> Iterators.product(range(xlim..., length=30), range(ylim..., length=30)) collect vec;
    x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
    d = size(x, 1)
    xj = unsqueeze(x, 1);
    n = norm2(xj, dims=2)
    n = n ./ maximum(n)
    @> n vec minimum, maximum, mean, std
    t = fill!(similar(xj), 0.1)
    # t = 0.5f0n
    # t = repeat(t, outer=(1,2,1))
    σ_t = marginal_prob_std(t)
    @≥ x unsqueeze(2) repeat(1, d, 1)
    J = @> unet(x, t)[:, 2, :]
    x = @> squeeze(xj, 1)
    @≥ J, x cpu.()
    x, y = eachrow(x);
    u, v = eachrow(0.05J);
    _, _, _, s = @> norm2(J, dims=1) maximum, minimum, mean, std

    pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
    arrow0!.(x, y, u, v; as=1.0, lw=1.0);
    # quiver!(x, y, quiver=(u, v), aspect_ratio=:equal)
    @> Plots.plot(pl_data, pl_perturbed_data, pl_sm_data, pl_r2, pl_oracle, pl_gradient; xlim, ylim, size=(1000, 800)) savefig("fig/$datetime_prefix-2d.png")
end

xlim = ylim = (-5, 5)
plot_data_gradients(normal_df, perturbed_df, regressor, oracle, unet, σX, μX; xlim, ylim)

