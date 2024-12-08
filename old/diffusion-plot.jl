include("diffusion-mnist.jl")
include("lib/utils.jl")

using Images
using ProgressMeter
using Plots, JLD2


diffusion_coeff(t, sigma=convert(eltype(t), 25.0f0)) = sigma .^ t

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

function convert_to_animation(x)
    frames = size(x)[end]
    batches = size(x)[end-1]
    animation = @animate for i = 1:frames+frames÷4
        if i <= frames
            heatmap(
                convert_to_image(x[:, :, :, :, i], batches),
                title="Iteration: $i out of $frames"
            )
        else
            heatmap(
                convert_to_image(x[:, :, :, :, end], batches),
                title="Iteration: $frames out of $frames"
            )
        end
    end
    return animation
end

"""
Helper function that generates inputs to a sampler.
num_images=5
num_steps=500
ϵ=1.0f-3
"""
function setup_sampler(device=gpu, num_images=5, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images)
    init_x = @> randn(Float32, (28, 28, 1, num_images)) .* expand_dims(marginal_prob_std(t), 3)
    @≥ t, init_x device.()
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

"""
Sample from a diffusion model using the Euler-Maruyama method.
# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
Δx = [f(x,t) - g²(t)s_θ(x,t)]dt + g(t)ΔW(t), where ΔW(t) ~ N(0, Δt)
"""
function Euler_Maruyama_sampler(unet, init_x, time_steps, Δt, device=gpu)
    x = mean_x = init_x
    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = diffusion_coeff(batch_time_step)
        mean_x = x .+ expand_dims(g, 3) .^ 2 .* unet(x, batch_time_step) .* Δt
        x = mean_x .+ sqrt(Δt) .* expand_dims(g, 3) .* device(randn(Float32, size(x)))
    end
    return mean_x
end

function plot_result(unet, args)
    args.seed > 0 && Random.seed!(args.seed)
    device = gpu
    unet = unet |> device
    time_steps, Δt, init_x = setup_sampler(device)
    # Euler-Maruyama
    euler_maruyama = Euler_Maruyama_sampler(unet, init_x, time_steps, Δt)
    sampled_noise = convert_to_image(init_x, size(init_x)[end])
    save(joinpath(args.save_path, "sampled_noise.jpeg"), sampled_noise)
    em_images = convert_to_image(euler_maruyama, size(euler_maruyama)[end])
    save(joinpath(args.save_path, "em_images.jpeg"), em_images)
end

BSON.@load "output/diffusion-mnist&batchsize=32&eta=0.0001&epoch=2(2).bson" unet args
unet = unet |> gpu

plot_result(unet, args)
