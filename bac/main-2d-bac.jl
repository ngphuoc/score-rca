using Revise
include("./imports.jl")
include("./rca.jl")
include("./rca-data.jl")
include("./lib/diffusion.jl")
include("./random-graph-datasets.jl")
include("./mlp-unet-2d.jl")

datetime_prefix = @> string(now())[1:16] replace(":"=>"h")

args = @env begin
    # Denoising
    input_dim = 2
    output_dim = 2
    hidden_dims = [50, 10]
    hidden_dim = 32  # hiddensize factor
    n_layers = 3
    embed_dim = 50  # hiddensize factor
    activation=swish
    perturbed_scale = 1f0
    fourier_scale=30.0f0
    # scale=25.0f0  # RandomFourierFeatures scale
    # σ_max=25f0
    σ_max = 6f0  # μ + 3σ pairwise Euclidean distances of input
    σ_min = 1f-3
    lr_regressor = 1e-3  # learning rate
    lr_unet = 1e-4  # learning rate
    decay = 1e-5  # weight decay parameter for AdamW
    to_device = Flux.gpu
    batchsize = 32
    epochs = 300
    save_path = ""
    load_path = "data/exp2d-joint.bson"
    # RCA
    min_depth = 2  # minimum depth of ancestors for the target node
    n_root_nodes = 1  # n_root_nodes
    n_timesteps = 40
    n_samples = 1000  # n observations
    n_anomaly_nodes = 2
    n_anomaly_samples = 100  # n faulty observations
    has_node_outliers = true  # node outlier setting
    n_reference_samples = 8  # n reference observations to calculate grad and shapley values, if n_reference_samples == 1 then use zero reference
    noise_scale = 1.0
    seed = 1  #  random seed
end

@info "Data"
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

@info "Model"
unet = MlpUnet(; args)
df = normal_df
unet = train_unet(unet, df; args)

@info "Plots"
include("plot-joint.jl")
xlim = ylim = (-5, 5)
plot_gradients(normal_df, perturbed_df, unet; xlim, ylim, args)

