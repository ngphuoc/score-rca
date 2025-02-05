include("./lib/utils.jl")
include("./lib/nn.jl")
include("./lib/nnlib.jl")
include("./models/toy.jl")
include("./dlsm-losses.jl")
include("./nmoon.jl")
using Flux
using Flux: crossentropy
using Flux.Data: DataLoader

args = @env begin
    # training =
    n_iters = 40001
    n_data = 10000
    batchsize = 1000
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
    λ_dsm = 2
    λ_dlsm = 2
    λ_ce = 0
    λ = 1.0  # coef: 1 (ce) 1 (dlsm) 0.125 (total)
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

function get_dataset()
    X, y = nmoons(Float64, n_data, 2, ε=0.25, d=2, repulse=(-0.25,0.0))
    DataLoader((X, y), batchsize=args.batchsize, shuffle=true)
end

ds = get_dataset()
(batch, labels) = ds |> first |> gpu
classifier_model = NCClassifier(; args) |> to_device;
score_model = NCScore(; args) |> to_device;

# classifier_loss(classifier_model, score_model, batch, labels)

classifier_loss = get_classifier_loss(σ_max, σ_min, λ_dlsm, 1)
opts = Flux.setup.([Adam(lr)], (classifier_model, score_model));

for iter = 1:20
    total_loss = 0.0
    for (batch, labels) = ds
        @≥ batch, labels gpu.()

        loss, (grads,) = Flux.withgradient((classifier_model, score_model), ) do (classifier_model, score_model)
            classifier_loss(classifier_model, score_model, batch, labels)
        end

        Flux.update!.(opts, (classifier_model, score_model), grads)
        total_loss += loss
    end
    @show iter, total_loss/length(ds)
end

classifier_loss = get_classifier_loss(σ_max, σ_min, λ_dlsm, 0)
opts = Flux.setup.([Adam(lr)], (classifier_model, score_model));

for iter = 1:20
    total_loss = 0.0
    for (batch, labels) = ds
        @≥ batch, labels gpu.()

        loss, (grads,) = Flux.withgradient((classifier_model, score_model), ) do (classifier_model, score_model)
            classifier_loss(classifier_model, score_model, batch, labels)
        end

        Flux.update!.(opts, (classifier_model, score_model), grads)
        total_loss += loss
    end
    @show iter, total_loss/length(ds)
end

