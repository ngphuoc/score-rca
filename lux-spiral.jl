using ArgCheck, CairoMakie, ConcreteStructs, Comonicon, DataAugmentation, DataDeps, FileIO, ImageCore, JLD2, Lux, LuxCUDA, MLUtils, Optimisers, ParameterSchedulers, ProgressBars, Random, Setfield, StableRNGs, Statistics, Zygote
using CUDA
using TensorBoardLogger: TBLogger, log_value, log_images
include("lib/utils.jl")

function sinusoidal_embedding(x::AbstractArray{T, 2}, min_freq::T, max_freq::T,
        embedding_dims::Int) where {T <: AbstractFloat}
    lower, upper = log(min_freq), log(max_freq)
    n = embedding_dims ÷ 2
    d = (upper - lower) / (n - 1)
    freqs = reshape(exp.(lower:d:upper) |> get_device(x), n, 1)
    x_ = 2 .* x .* freqs
    return cat(sinpi.(x_), cospi.(x_); dims=Val(1))
end

function residual_block(in_channels::Int, out_channels::Int)
    return Parallel(+,
        in_channels == out_channels ? NoOpLayer() :
        Dense(in_channels => out_channels),
        Chain(BatchNorm(in_channels; affine=false),
            Dense(in_channels => out_channels, swish),
            Dense(out_channels => out_channels));
        name="ResidualBlock(in_chs=$in_channels, out_chs=$out_channels)")
end

function downsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
    # @show in_channels out_channels block_depth
    return @compact(;
        name="DownsampleBlock(in_chs=$in_channels, out_chs=$out_channels, block_depth=$block_depth)",
        residual_blocks=Tuple(residual_block(
            ifelse(i == 1, in_channels, out_channels), out_channels)
                              for i in 1:block_depth),
        block_depth
    ) do x
        skips = (x,)
        for i in 1:block_depth
            skips = (skips..., residual_blocks[i](last(skips)))
        end
        y = last(skips)
        @return y, skips
    end
end

function upsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
    return @compact(;
        name="UpsampleBlock(in_chs=$in_channels, out_chs=$out_channels, block_depth=$block_depth)",
        residual_blocks=Tuple(residual_block(
            ifelse(i == 1, in_channels + out_channels, out_channels * 2), out_channels)
                              for i in 1:block_depth),
        block_depth
    ) do x_skips
        x, skips = x_skips
        # x = upsample(x)
        for i in 1:block_depth
            x = residual_blocks[i](cat(x, skips[end-i+1]; dims=Val(1)))
        end
        @return x
    end
end

function unet_model(; channels=[32, 64, 96, 128], block_depth=2, min_freq=1.0f0, max_freq=1000.0f0, embedding_dims=32)
    conv_in = Dense(2 => channels[1])
    conv_out = Dense(channels[1] => 2; init_weight=Lux.zeros32)
    channel_input = embedding_dims + channels[1]
    down_blocks = [downsample_block((i == 1 ? channel_input : channels[i-1]), channels[i], block_depth)
                   for i in 1:(length(channels)-1)]
    residual_blocks = Chain([residual_block(
                             ifelse(i == 1, channels[end - 1], channels[end]), channels[end])
                             for i in 1:block_depth]...)
    reverse!(channels)
    up_blocks = [upsample_block(in_chs, out_chs, block_depth)
                 for (in_chs, out_chs) in zip(channels[1:(end - 1)], channels[2:end])]

    #! format: off
    return @compact(;
        conv_in, conv_out, down_blocks, residual_blocks, up_blocks,
        min_freq, max_freq, embedding_dims,
        num_blocks=(length(channels) - 1)
    ) do x_t::Tuple{AbstractArray{<:Real, 2}, AbstractArray{<:Real, 2}}
    #! format: on
        x, t = x_t
        emb = sinusoidal_embedding(t, min_freq, max_freq, embedding_dims)
        x = cat(conv_in(x), emb; dims=Val(1))
        skips_at_each_stage = ()
        for i in 1:num_blocks
            x, skips = down_blocks[i](x)
            skips_at_each_stage = (skips_at_each_stage..., skips)
        end
        x = residual_blocks(x)
        for i in 1:num_blocks
            x = up_blocks[i]((x, skips_at_each_stage[end - i + 1]))
        end
        @return conv_out(x)
    end
end

expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))

function sde(rng::AbstractRNG; channels, block_depth, min_freq, max_freq, σ_max, embedding_dims)
    unet = unet_model(; channels, block_depth, min_freq, max_freq, embedding_dims)
    bn = BatchNorm(2; affine=false, track_stats=true)
    σ_max = σ_max

    return @compact(; unet, bn, rng, σ_max, dispatch=:SDE
    ) do x::AbstractArray{<:Real, 2}
        x_bn = bn(x)
        rng = Lux.replicate(rng)
        z = rand_like(rng, x_bn)
        t = rand_like(rng, x_bn, (1, size(x_bn, 2))).* (1f0 - 1f-5) .+ 1f-5
        σ_t = marginal_prob_std(t; σ_max)
        x̃ = @. x + z * σ_t
        score = unet((x̃, t)) ./ σ_t  # parametrize the normalized score
        x̂ = x̃ + score .* σ_t
        @return (score, σ_t, z, x̂)
    end
end

marginal_prob_std(t; σ_max) = sqrt.((σ_max .^ (2t) .- 1.0f0) ./ 2.0f0 ./ log(σ_max))

diffusion_coeff(t, σ_max=convert(eltype(t), 25.0f0)) = σ_max .^ t

# function Euler_Maruyama_sampler(model, init_x::AbstractArray{T,N}, time_steps, Δt, σ_max) where {T,N}
#     x = mean_x = init_x
#     @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
#         batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
#         g = @> diffusion_coeff(batch_time_step, σ_max) expand_dims(N-1)
#         mean_x = x .+ g .^ 2 .* model(x, batch_time_step) .* Δt
#         x = mean_x .+ g .* sqrt(Δt) .* oftype(init_x, randn(Float32, size(x)))
#     end
#     return mean_x
# end

function score_matching_loss(unet, x::AbstractArray{T,N}; ϵ=1.0f-5, σ_max=25f0) where {T,N}
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1.0f0 - ϵ) .+ ϵ;
    z = randn!(similar(x))
    σ_t = expand_dims(marginal_prob_std(t; σ_max), N-1)
    # (batch) of perturbed 𝘹(𝘵)'s to approximate 𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0))
    x̃ = x + z .* σ_t
    score = unet(x̃, t)
    sum(abs2, score .* σ_t + z) / batchsize
end

function sm_loss(unet, x::AbstractMatrix{<:Real}; ϵ=1.0f-5, σ_max=25f0)
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    z = randn!(similar(x))
    σ_t = expand_dims(marginal_prob_std(t; σ_max), 1)
    x̃ = x + z .* σ_t
    score = unet(x̃, t)
    return sum(abs2, score .* σ_t + z) / batchsize
end

const maeloss = MAELoss()

"""
sde model's loss function, data at 4th position
"""
function loss_function(model, ps, st, x)
    (score, σ_t, z, x̂), st = Lux.apply(model, x, ps, st)
    score_loss = sum(abs2, score .* σ_t + z) / size(z)[end]
    image_loss = maeloss(x̂, x)
    return score_loss, st, (; image_loss, score_loss)
end


function make_spiral(rng::AbstractRNG, n_samples::Int=1000)
    t_min = 1.5π
    t_max = 4.5π
    t = rand(rng, n_samples) * (t_max - t_min) .+ t_min
    x = t .* cos.(t)
    y = t .* sin.(t)
    [x y]'
end

function main()
    args = @env begin
        epochs = 100
        image_size = 128
        batchsize = 32
        σ_max = 25f0
        learning_rate_start = 1.0f-3
        learning_rate_end = 1.0f-5
        weight_decay = 1.0f-6
        checkpoint_interval = 25
        diffusion_steps = 80
        generate_image_interval = 5
        # model hyper params
        channels = [32, 64, 96, 128]
        block_depth = 2
        min_freq = 1.0f0
        max_freq = 1000.0f0
        embedding_dims = 32
        max_signal_rate = 0.95f0
        generate_image_seed = 12
        # inference specific
        inference_mode = false
        saved_model_path = nothing
        generate_n_images = 12
    end

    rng = Random.default_rng()
    Random.seed!(rng, 1234)
    gdev = gpu_device()
    @info "Using device: $gdev"

    @info "Building model"
    model = sde(rng; args.channels, args.block_depth, args.min_freq, args.max_freq, args.σ_max, args.embedding_dims)
    ps, st = Lux.setup(rng, model) |> gdev
    num_params = Lux.parameterlength(ps)
    println("Number of model parameters: ", num_params)  # 1950627

    @info "Preparing dataset"
    X = @> make_spiral(rng) f32
    n = size(X, 2)
    data_loader = DataLoader((X,); args.batchsize, collate=true) |> gdev
    (x,) = @> data_loader first
    l, = loss_function(model, ps, st, x)  # check clip_var
    @show l

    @info "Training"
    tstate = Training.TrainState(
        model, ps, st, AdamW(; eta=args.learning_rate_start, lambda=args.weight_decay))
    scheduler = CosAnneal(args.learning_rate_start, args.learning_rate_end, args.epochs)

    for epoch in 1:args.epochs
        pbar = ProgressBar(data_loader)
        eta = scheduler(epoch)
        tstate = Optimisers.adjust!(tstate, eta)
        image_losses = Vector{Float32}(undef, length(data_loader))
        score_losses = Vector{Float32}(undef, length(data_loader))
        for (i, x) in enumerate(data_loader)
            (_, _, stats, tstate) = Training.single_train_step!(
                AutoZygote(), loss_function, x, tstate)
            image_losses[i] = stats.image_loss
            score_losses[i] = stats.score_loss
            ProgressBars.update(pbar)
            set_description(
                pbar, "Epoch: $(epoch) Image Loss: $(mean(view(image_losses, 1:i))) Score \
                       Loss: $(mean(view(score_losses, 1:i)))")
        end
    end
    return tstate
end

main()

