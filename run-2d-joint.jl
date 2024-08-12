using Optimisers, BSON
using ProgressMeter: Progress, next!
include("./data.jl")
include("./lib/diffusion.jl")

args = @env begin
    input_dim = 2
    output_dim = 2
    hidden_dim = 32  # hiddensize factor
    embed_dim = 32  # hiddensize factor
    scale=30.0f0  # RandomFourierFeatures scale
    σ_max = 5.0
    σ_min = 1e-3
    lr = 3e-4
    to_device = gpu
    batchsize = 32
    epochs = 300
    save_path = "output"
    model_file = "2d-joint.bson"
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
        linear0 = Dense(E, 8H),
        linear1 = Dense(E, 4H),
        linear2 = Dense(E, 2H),
        linear3 = Dense(E, 1H),
        dense0 = Dense(I, 8H, relu),
        dense1 = Dense(8H, 4H, relu),
        dense2 = Dense(4H, 2H, relu),
        dense3 = Dense(2H, 1H),
        out = Dense(1H, O),
       ))
end

function (m::ConditionalScore)(x::AbstractArray{T,N}, t) where {T,N}
    nn = m.layers
    embed = nn.embed(nn.gaussfourierproj(t))
    h0 = nn.dense0(x) .+ nn.linear0(embed)
    h1 = nn.dense1(h0) .+ nn.linear1(embed)
    h2 = nn.dense2(h1) .+ nn.linear2(embed)
    h3 = nn.dense3(h2) .+ nn.linear3(embed)
    h = nn.out(h3)
    h ./ expand_dims(marginal_prob_std(t, σ_max), N-1)
end

#--  fit score model

score_model = ConditionalScore(; args) |> to_device;
opt = Flux.setup(Optimisers.Adam(lr), score_model);
bn, X, Y = get_data()
loader = DataLoader((X, Y); args.batchsize, shuffle=true)
(x, y) = loader |> first
@≥ x, y gpu.()
xy = vcat(x, y)

diffusion_loss(score_model, xy)

for epoch = 1:args.epochs
    # progress = Progress(length(loader))
    total_loss = 0.0
    for (x, y) = loader
        @≥ x, y gpu.()
        xy = vcat(x, y)
        loss, (grad,) = Flux.withgradient(score_model, ) do score_model
            diffusion_loss(score_model, xy)
        end
        Flux.update!(opt, score_model, grad)
        total_loss += loss
        # next!(progress; showvalues=[(:loss, loss)])
    end
    epoch % 10 == 0 && @show epoch, total_loss/length(loader)
end

#--  sampling

"""
Helper function that generates inputs to a sampler.
n=100
num_steps=500
ϵ=1.0f-3
"""
function setup_sampler(device, n=100, num_steps=500, ϵ=1.0f-3; args)
    t = ones(Float32, n)
    init_x = @> randn(Float32, (args.input_dim, n)) .* expand_dims(marginal_prob_std(t, σ_max), 1) device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

function plot_result(score_model, args)
    device = gpu
    score_model = score_model |> device
    time_steps, Δt, init_x = setup_sampler(device; args)

    function plot2d(xy, xlims=(-5,5), ylims=(-5,5))
        x, y = @> xy cpu eachrow
        pl = scatter(x, y; aspect_ratio=:equal, xlab="x", ylab="y", xlims, ylims)
    end

    # Euler-Maruyama
    @> plot2d(vcat(X, Y)) savefig(joinpath(args.save_path, "2d-joint-data.png"))
    @> plot2d(init_x) savefig(joinpath(args.save_path, "2d-joint-sampled_noise.png"))
    euler_maruyama = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt, σ_max)
    @> plot2d(euler_maruyama) savefig(joinpath(args.save_path, "2d-joint-em_images.png"))

end

model_path = joinpath(args.save_path, args.model_file)
let score_model = cpu(score_model), args=args
    BSON.@save model_path score_model args
end
BSON.@load model_path score_model args
score_model = score_model |> gpu

plot_result(score_model, args)

