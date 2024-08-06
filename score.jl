using Flux
using Flux: crossentropy
include("./data.jl")

args = @env begin
    # training =
    n_iters = 40001
    batch_size = 2000
    snapshot_freq = 5000
    log_freq = 500
    loss_type = "ce"  # loss = "dlsm" "total"

    # sampling =
    sampling_type = "vesde_pc_sampler"
    noise_σ = 1.0
    num_steps = 1000  # num_steps = 50 (score)
    epsilon = 0.01
    width = 40
    height = 25
    density = 35

    # data =
    dataset = "inter_twinning_moon"

    # model
    type = "score_model"
    nf = 16
    σ = 7.5
    σ_max = 10.0
    σ_min = 1e-3
    classes = 2
    weighting_dsm = 2
    weighting_dlsm = 2
    weighting_ce = 0
    coef = 1.0  # coef: 1 (ce) 1 (dlsm) 0.125 (total)
    scaling_factor = 1

    # optim =
    weight_decay = 0
    lr = 5e-4  # lr = 1e-5 (ce) 2e-5 (dlsm) 6.5e-4 (score)
    beta1 = 0.9
    eps = 1e-8

    eval_type = "sampling"
    seed = 1
    to_device = gpu
end

# Initialize the classifier and the score model.
classifier_model = NCClassifier(; args) |> to_device
score_model = NCScore(; args) |> to_device
classifier_loss = get_classifier_loss((σ_max, σ_min), loss_type, weighting_dlsm, weighting_ce, coef)

opts = Flux.setup.(Adam(lr), (classifier_model, score_model));

ds = get_dataset(args)

for (batch, labels) in ds
    @≥ batch, labels gpu.()

    loss, grads = Flux.withgradient((classifier_model, score_model)) do (classifier_model, score_model)
        classifier_loss(classifier_model, score_model, batch, labels)
    end

    Flux.update!.(opts, (classifier_model, score_model), grads)
    @show loss
end

