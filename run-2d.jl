include("./data.jl")
using DifferentialEquations

args = @env begin
    input_dim = 1
    hidden_dim = 2  # hiddensize factor
    σ_max = 10.0
    σ_min = 1e-3
    lr = 3e-4
    to_device = gpu
    batchsize = 100
end

struct ConditionalScore
    layers
    cond_layers
    linear
end
@functor ConditionalScore layers, cond_layers, linear

"""noise-conditioned score model"""
function ConditionalScore(; args)
    hidden_dim = args.hidden_dim
    layers = Chain(
                   Dense(input_dim, hidden_dim * 16, relu),
                   Dense(hidden_dim * 16, hidden_dim * 8, relu),
                   Dense(hidden_dim * 8, hidden_dim * 4)
                  )
    cond_layers = Chain(
                        Dense(1, hidden_dim * 4, relu),
                        Dense(hidden_dim * 4, hidden_dim * 2, relu),
                        Dense(hidden_dim * 2, hidden_dim * 1)
                       )
    linear = Dense(hidden_dim * 5, 2)
    ConditionalScore(layers, cond_layers, linear)
end

function (m::ConditionalScore)(x, σ)
    @unpack layers, cond_layers, linear = m
    σ = cond_layers(σ)
    x = layers(x)
    x_cond_cat = cat(x, σ, dims=1)
    out = linear(x_cond_cat)
    return out
end

function score_loss(score_model, batch)
    # Calculate standard deviation
    batchsize = size(batch)[end]
    t = rand(batchsize)
    σ = @> σ_min * (σ_max / σ_min) .^ t unsqueeze(1) to_device
    z = randn(size(batch)) |> to_device
    perturbed_batch = @. batch + σ * z
    score = score_model(perturbed_batch, σ)
    l = sum(abs2, σ .* score .+ z) / batchsize
    return l
end

#--  fit score model

score_model = ConditionalScore(; args) |> to_device;
opt = Flux.setup(Adam(lr), score_model);
ds = DataLoader((X, y); args.batchsize, shuffle=true)
(batch, labels) = ds |> first
@≥ batch, labels gpu.()
score_loss(score_model, batch)

for iter = 1:100
    total_loss = 0.0
    for (batch, labels) = ds
        @≥ batch, labels gpu.()
        loss, (grad,) = Flux.withgradient(score_model, ) do score_model
            score_loss(score_model, batch)
        end
        Flux.update!(opt, score_model, grad)
        total_loss += loss
    end
    @show iter, total_loss/length(ds)
end

#--  sampling

diffusion_coeff(t, sigma=convert(eltype(t), 25.0f0)) = sigma .^ t

"""
Helper function that generates inputs to a sampler.
"""
function setup_sampler(device, num_images=5, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images) |> device
    init_x = (
        randn(Float32, (28, 28, 1, num_images)) .*
        expand_dims(marginal_prob_std(t), 3)
    ) |> device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

"""
Sample from a diffusion model using the Euler-Maruyama method.

# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    x = mean_x = init_x
    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = diffusion_coeff(batch_time_step)
        mean_x = x .+ expand_dims(g, 3) .^ 2 .* model(x, batch_time_step) .* Δt
        x = mean_x .+ sqrt(Δt) .* expand_dims(g, 3) .* randn(Float32, size(x))
    end
    return mean_x
end

