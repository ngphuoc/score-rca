using ArgCheck, CairoMakie, ConcreteStructs, Comonicon, DataAugmentation, DataDeps, FileIO, ImageCore, JLD2, Lux, LuxCUDA, MLUtils, Optimisers, ParameterSchedulers, ProgressBars, Random, Setfield, StableRNGs, Statistics, Zygote
using CUDA
using TensorBoardLogger: TBLogger, log_value, log_images

include("spiral-data.jl")

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
        Chain(BatchNorm(out_channels; affine=false),
            Dense(in_channels => out_channels, swish),
            Dense(out_channels => out_channels));
        name="ResidualBlock(in_chs=$in_channels, out_chs=$out_channels)")
end

function downsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
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

function unet_model(image_size::Tuple{Int, Int}; channels=[32, 64, 96, 128],
        block_depth=2, min_freq=1.0f0, max_freq=1000.0f0, embedding_dims=32)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Dense(2 => channels[1])
    conv_out = Dense(channels[1] => 2; init_weight=Lux.zeros32)

    channel_input = embedding_dims + channels[1]
    down_blocks = [downsample_block(
                       ifelse(i == 1, channel_input, channels[i-1]), channels[i], block_depth)
                       for i in 1:(length(channels) - 1)]
    residual_blocks = Chain([residual_block(
                             ifelse(i == 1, channels[end - 1], channels[end]), channels[end])
                             for i in 1:block_depth]...)
    reverse!(channels)
    up_blocks = [upsample_block(in_chs, out_chs, block_depth)
                 for (in_chs, out_chs) in zip(channels[1:(end - 1)], channels[2:end])]

    #! format: off
    return @compact(;
        upsample, conv_in, conv_out, down_blocks, residual_blocks, up_blocks,
        min_freq, max_freq, embedding_dims,
        num_blocks=(length(channels) - 1)
    ) do x::Tuple{AbstractArray{<:Real, 4}, AbstractArray{<:Real, 4}}
    #! format: on
        noisy_images, noise_variances = x

        emb = upsample(sinusoidal_embedding(
            noise_variances, min_freq, max_freq, embedding_dims))
        x = cat(conv_in(noisy_images), emb; dims=Val(1))
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

function sde(rng::AbstractRNG, args...; min_signal_rate=0.02f0,
        max_signal_rate=0.95f0, kwargs...)
    unet = unet_model(args...; kwargs...)
    bn = BatchNorm(2; affine=false, track_stats=true)

    return @compact(; unet, bn, rng, min_signal_rate,
        max_signal_rate, dispatch=:SDE
    ) do x::AbstractArray{<:Real, 2}
        images = bn(x)
        rng = Lux.replicate(rng)

        noises = rand_like(rng, images)
        diffusion_times = rand_like(rng, images, (1, size(images, 2)))

        noise_rates, signal_rates = diffusion_schedules(
            diffusion_times, min_signal_rate, max_signal_rate)

        noisy_images = @. signal_rates * images + noise_rates * noises

        pred_noises, pred_images = denoise(unet, noisy_images, noise_rates, signal_rates)

        @return noises, images, pred_noises, pred_images
    end
end

