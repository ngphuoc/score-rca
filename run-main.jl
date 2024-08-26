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
    σ_max = 5.0
    σ_min = 1e-3
    lr_regressor = 1e-3  # learning rate
    lr_unet = 1e-3  # learning rate
    decay = 1e-5  # weight decay parameter for AdamW
    to_device = Flux.gpu
    batchsize = 32
    epochs = 50
    save_path = "data/exp2d.bson"
    load_path = ""
    pkl_path = "data/exp2d.pkl"
    # RCA
    min_depth = 2  # minimum depth of ancestors for the target node
    n_root_nodes = 1  # n_root_nodes
    n_samples = 500  # n observations
    n_anomaly_nodes = 2
    n_outlier_samples = 8  # n faulty observations
    has_node_outliers = true  # node outlier setting
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    noise_scale = 1.0
    seed = 1  #  random seed
end

@info "Creating new data and training new models"

n_nodes = round(Int, min_depth^2 / 3 + 1)
n_root_nodes = 1
n_downstream_nodes = n_nodes - n_root_nodes
scale, hidden = 0.2, 2
dag = random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, scale, hidden)
train_df, anomaly_df = draw_normal_anomaly(dag; args)

μX, σX = @> X mean(dims=2), std(dims=2);

#-- train models
hidden_dims = [2, ]
paj_mask = B = @> adjacency_matrix(g0) Matrix{Bool}
@info "Creating unet and mlp models"
regressor = GroupMlpRegression(B; μX, σX, hidden_dims, activation=Flux.tanh)
unet = GroupMlpUnet(B)

regressor, unet = train_score_model_from_ground_truth_dag(regressor, unet, train_df; args)

