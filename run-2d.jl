using Optimisers, BSON
using ProgressMeter: Progress, next!
include("./data.jl")
include("./lib/diffusion.jl")

args = @env begin
    input_dim = 1
    output_dim = 1
    hidden_dim = 8  # hiddensize factor
    embed_dim = 16  # hiddensize factor
    scale=30.0f0  # RandomFourierFeatures scale
    σ_max = 10.0
    σ_min = 1e-3
    lr = 3e-4
    to_device = gpu
    batchsize = 100
    epochs = 100
end

struct ConditionalScore
    layers::NamedTuple
end
@functor ConditionalScore

"""noise-conditioned score model"""
function ConditionalScore(; args)
    I, O, E, H  = (args.input_dim, args.output_dim, args.embed_dim, args.hidden_dim)
    return ConditionalScore((
        gaussfourierproj = RandomFourierFeatures(E, scale),
        embed = Dense(E, E, swish),
        linear1 = Dense(E, 4H),
        linear2 = Dense(E, 2H),
        linear3 = Dense(E, 1H),
        dense1 = Dense(I, 4H, relu),
        dense2 = Dense(4H, 2H, relu),
        dense3 = Dense(2H, 1H),
        out = Dense(1H, 1),
       ))
end

function (m::ConditionalScore)(x::AbstractArray{T,N}, t) where {T,N}
    nn = m.layers
    embed = nn.embed(nn.gaussfourierproj(t))
    h1 = nn.dense1(x) .+ nn.linear1(embed)
    h2 = nn.dense2(h1) .+ nn.linear2(embed)
    h3 = nn.dense3(h2) .+ nn.linear3(embed)
    h = nn.out(h3)
    h ./ expand_dims(marginal_prob_std(t), N-1)
end

#--  fit score model

score_model = ConditionalScore(; args) |> to_device;
opt = Flux.setup(Optimisers.Adam(lr), score_model);
loader = DataLoader((X, y); args.batchsize, shuffle=true)
(x, y) = loader |> first
@≥ x, y gpu.()

diffusion_loss(score_model, x)

for epoch = 1:args.epochs
    @info "Epoch $(epoch)"
    progress = Progress(length(loader))
    total_loss = 0.0
    for (x, y) = loader
        @≥ x, y gpu.()
        loss, (grad,) = Flux.withgradient(score_model, ) do score_model
            diffusion_loss(score_model, x)
        end
        Flux.update!(opt, score_model, grad)
        total_loss += loss
        next!(progress; showvalues=[(:loss, loss)])
    end
    @show epoch, total_loss/length(loader)
end

#--  sampling

"""
Helper function that generates inputs to a sampler.
n=100
num_steps=500
ϵ=1.0f-3
"""
function setup_sampler(device, n=100, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, n)
    init_x = @> randn(Float32, (1, n)) .* expand_dims(marginal_prob_std(t), 1) device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

function plot_result(model, args)
    device = gpu
    model = model |> device
    time_steps, Δt, init_x = setup_sampler(device)
    # Euler-Maruyama
    euler_maruyama = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    sampled_noise = convert_to_image(init_x, size(init_x)[end])

    save(joinpath(args.save_path, "2d-sampled_noise.jpeg"), sampled_noise)
    em_images = convert_to_image(euler_maruyama, size(euler_maruyama)[end])
    save(joinpath(args.save_path, "2d-em_images.jpeg"), em_images)
end

let score_model = cpu(score_model)
    BSON.@save "output/2d.bson" score_model args
end

BSON.@load "output/2d.bson" score_model args
score_model = score_model |> gpu

plot_result(score_model, args)

