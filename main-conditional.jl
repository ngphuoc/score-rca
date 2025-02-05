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


include("run-plot.jl")
xlim = ylim = (-5, 5)
plot_data_gradients(normal_df, perturbed_df, regressor, oracle, unet, σX, μX; xlim, ylim)

