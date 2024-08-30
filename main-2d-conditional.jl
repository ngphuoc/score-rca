using Optimisers, BSON
using ProgressMeter: Progress, next!
include("./data.jl")
include("./lib/diffusion.jl")

args = @env begin
    hidden_dim = 16  # hiddensize factor
    embed_dim = 16  # hiddensize factor
    scale=10.0f0  # RandomFourierFeatures scale
    σ_max = 5.0
    lr = 3e-4
    to_device = gpu
    batchsize = 32
    epochs = 300
    save_path = "output"
    model_file = "2d-conditional.bson"
end

struct ConditionalScore
    layers::NamedTuple
end
@functor ConditionalScore

"""noise-conditioned score model"""
function ConditionalScore(input_dim; args)
    E, H  = (args.embed_dim, args.hidden_dim)
    return ConditionalScore((
        gaussfourierproj = RandomFourierFeatures(E, scale),
        embed = Dense(E, E, swish),
        ft0 = Dense(E, 8H),
        ft1 = Dense(E, 4H),
        ft2 = Dense(E, 2H),
        ft3 = Dense(E, 1H),
        fy0 = Dense(1, 8H, relu),
        fy1 = Dense(8H, 4H, relu),
        fy2 = Dense(4H, 2H, relu),
        fy3 = Dense(2H, 1H),
        fx0 = Dense(input_dim, 8H, relu),
        # fx1 = Dense(8H, 4H, relu),
        # fx2 = Dense(4H, 2H, relu),
        # fx3 = Dense(2H, 1H),
        out = Dense(1H, 1),
       ))
end

function (m::ConditionalScore)(y, x, t)
    nn = m.layers
    embed = nn.embed(nn.gaussfourierproj(t))
    h0 = nn.fy0(y) .+ nn.fx0(x) .+ nn.ft0(embed)
    h1 = nn.fy1(h0) .+ nn.ft1(embed)
    h2 = nn.fy2(h1) .+ nn.ft2(embed)
    h3 = nn.fy3(h2) .+ nn.ft3(embed)
    h = nn.out(h3)
    h ./ expand_dims(marginal_prob_std(t, σ_max), N-1)
end

#--  fit score model

input_dim = 1  # num parents
score_model = ConditionalScore(input_dim; args) |> to_device;
opt = Flux.setup(Optimisers.Adam(lr), score_model);
bn, X, Y = get_data()
bn.cpds[2]
BayesNets.name.(bn.cpds)
f(x; device=gpu) = gpu(bn[:Y].a) .* x
loader = DataLoader((X, Y); args.batchsize, shuffle=true)
(x, y) = loader |> first
@≥ x, y gpu.()

conditional_score_matching_loss(score_model, f, y, x)

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

