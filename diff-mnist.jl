using MLDatasets
using Functors: @functor
using Flux
using Flux: chunk, params
using MLUtils
using Parameters: @with_kw
using BSON
using CUDA
using Images
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
using Statistics
using Dates
using JLD2
# include("../lib/utils.jl")
# include("../lib/diffusion.jl")
include("./lib/utils.jl")
include("./lib/diffusion.jl")

struct UNet
    layers::NamedTuple
end

@functor UNet

function UNet(c, channels = [32, 64, 128, 256], embed_dim = 256, scale = 30.0f0)
    k = 10  # number of classes
    bias = false
    return UNet((
        gaussfourierproj = RandomFourierFeatures(embed_dim, scale),
        linear = Dense(embed_dim, embed_dim, swish),
        # Encoding
        conv1 = Conv((3, 3), c => channels[1], stride = 1, pad = 1; bias),
        dense1 = Dense(embed_dim, channels[1]),
        gnorm1 = GroupNorm(channels[1], 4, swish),
        conv2 = Conv((3, 3), channels[1] => channels[2], stride = 2, pad = 1; bias),
        dense2 = Dense(embed_dim, channels[2]),
        gnorm2 = GroupNorm(channels[2], 32, swish),
        conv3 = Conv((3, 3), channels[2] => channels[3], stride = 2, pad = 1; bias),
        dense3 = Dense(embed_dim, channels[3]),
        gnorm3 = GroupNorm(channels[3], 32, swish),
        conv4 = Conv((3, 3), channels[3] => channels[4], stride = 2, pad = 1; bias),
        dense4 = Dense(embed_dim, channels[4]),
        gnorm4 = GroupNorm(channels[4], 32, swish),
        # Decoding
        tconv4 = ConvTranspose((3, 3), channels[4] => channels[3], stride = 2, pad = 1; bias),
        dense5 = Dense(embed_dim, channels[3]),
        tgnorm4 = GroupNorm(channels[3], 32, swish),
        tconv3 = ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad = (1, 0, 1, 0), stride = 2; bias),
        dense6 = Dense(embed_dim, channels[2]),
        tgnorm3 = GroupNorm(channels[2], 32, swish),
        tconv2 = ConvTranspose((3, 3), channels[2] + channels[2] => channels[1], pad = (1, 0, 1, 0), stride = 2; bias),
        dense7 = Dense(embed_dim, channels[1]),
        tgnorm2 = GroupNorm(channels[1], 32, swish),
        tconv1 = ConvTranspose((3, 3), channels[1] + channels[1] => 1, stride = 1, pad = 1; bias),
    ))
end

function (model::UNet)(x, t)
    # Embedding
    embed = model.layers.linear(model.layers.gaussfourierproj(t))
    # Encoder
    h1 = model.layers.conv1(x);
    h1 = h1 .+ expand_dims(model.layers.dense1(embed), 2);
    h1 = model.layers.gnorm1(h1);
    h2 = model.layers.conv2(h1);
    h2 = h2 .+ expand_dims(model.layers.dense2(embed), 2);
    h2 = model.layers.gnorm2(h2);
    h3 = model.layers.conv3(h2);
    h3 = h3 .+ expand_dims(model.layers.dense3(embed), 2);
    h3 = model.layers.gnorm3(h3);
    h4 = model.layers.conv4(h3);
    h4 = h4 .+ expand_dims(model.layers.dense4(embed), 2);
    h4 = model.layers.gnorm4(h4);
    # Decoder;
    h = model.layers.tconv4(h4);
    h = h .+ expand_dims(model.layers.dense5(embed), 2);
    h = model.layers.tgnorm4(h);
    h = model.layers.tconv3(cat(h, h3; dims = 3));
    h = h .+ expand_dims(model.layers.dense6(embed), 2);
    h = model.layers.tgnorm3(h);
    h = model.layers.tconv2(cat(h, h2, dims = 3));
    h = h .+ expand_dims(model.layers.dense7(embed), 2);
    h = model.layers.tgnorm2(h);
    h = model.layers.tconv1(cat(h, h1, dims = 3));
    # Scaling Factor;
    h ./ expand_dims(marginal_prob_std(t; σ_max), 3);
end

function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = reshape(xtrain, 28, 28, 1, :)
    DataLoader((xtrain, ytrain), batchsize = batch_size, shuffle = true)
end

function struct2dict(::Type{DT}, s) where {DT<:AbstractDict}
    DT(x => getfield(s, x) for x in fieldnames(typeof(s)))
end
struct2dict(s) = struct2dict(Dict, s)

# arguments for the `train` function
args = @env_const begin
    η = 1e-4                                        # learning rate
    batch_size = 32                                 # batch size
    epochs = 50                                     # number of epochs
    save_every = 2
    seed = 1                                        # random seed
    cuda = true                                    # use CPU
    verbose_freq = 10                               # logging for every verbose_freq iterations
    tblogger = true                                 # log training with tensorboard
    save_path = "output"                            # results path
    dryrun = false
    σ_max = 25f0
end

function train()
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        dev = gpu
        @info "Training on GPU"
    else
        dev = cpu
        @info "Training on CPU"
    end

    # load MNIST images
    loader = get_data(args.batch_size)
    model = UNet(1) |> dev
    opt_state = Flux.setup(Adam(), model);

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # # Training
    # epoch = 1
    # x, _ = loader |> first |> dev
    # score_matching_loss(model, x)
    # grads, = Flux.gradient(model) do model
    #     score_matching_loss(model, x)
    # end

    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x, _) = loader
            x = dev(x)
            loss, grads = Flux.withgradient(model) do model
                score_matching_loss(model, x)
            end
            Flux.update!(opt_state, model, grads[1])
            next!(progress; showvalues = [(:loss, loss)])

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss = loss
                end
            end
            train_steps += 1
            args.dryrun && break
        end
        args.dryrun && break
    end

    # save model
    epoch = args.epochs
    model_str = "&batchsize=$(args.batch_size)&eta=$(args.η)&epoch=$epoch($(args.epochs))"
    model_path = joinpath(args.save_path, "diffusion-mnist$model_str.bson")
    let model = cpu(model), args = struct2dict(args)
        BSON.@save model_path model args
        @info "Model saved: $(model_path)"
    end
end

# if abspath(PROGRAM_FILE) == @__FILE__
    train()
# end

