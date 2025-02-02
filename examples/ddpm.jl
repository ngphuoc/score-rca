using ArgParse
using Flux
using Random
using Statistics
using Plots

# Custom positional embedding implementation
struct PositionalEmbedding
    emb_size::Int
    emb_type::String
end

function (pe::PositionalEmbedding)(x)
    if pe.emb_type == "sinusoidal"
        t = (1:pe.emb_size)'
        freq = 1.0 ./ (10000.0 .^ ((t .- 1) ./ pe.emb_size))
        x = reshape(x, :, 1)
        return hcat(sin.(x .* freq), cos.(x .* freq))
    else
        error("Embedding type $(pe.emb_type) not implemented.")
    end
end

# Custom MLP structure
struct Block
    ff::Dense
end

function Block(size::Int)
    return Block(Dense(size, size, relu))
end

function (b::Block)(x)
    return x .+ b.ff(x)
end

struct MLP
    time_emb::PositionalEmbedding
    input_emb1::PositionalEmbedding
    input_emb2::PositionalEmbedding
    layers::Chain
end

function MLP(hidden_size::Int, hidden_layers::Int, emb_size::Int, time_emb::String, input_emb::String)
    time_emb = PositionalEmbedding(emb_size, time_emb)
    input_emb1 = PositionalEmbedding(emb_size, input_emb)
    input_emb2 = PositionalEmbedding(emb_size, input_emb)

    concat_size = emb_size * 3
    @show concat_size, hidden_size
    layer_list = Any[Dense(concat_size, hidden_size, relu)]
    for _ in 1:hidden_layers
        push!(layer_list, Block(hidden_size))
    end
    push!(layer_list, Dense(hidden_size, 2))

    return MLP(time_emb, input_emb1, input_emb2, Chain(layer_list...))
end

function (mlp::MLP)(x, t)
    x1_emb = mlp.input_emb1(x[:, 1])
    x2_emb = mlp.input_emb2(x[:, 2])
    t_emb = mlp.time_emb(t)
    @show size.((x1_emb, x2_emb, t_emb))
    x_combined = hcat(x1_emb, x2_emb, t_emb)
    return mlp.layers(x_combined')
end

# Noise Scheduler
struct NoiseScheduler
    num_timesteps::Int
    betas::Vector{Float32}
    alphas::Vector{Float32}
    alphas_cumprod::Vector{Float32}
    sqrt_alphas_cumprod::Vector{Float32}
    sqrt_one_minus_alphas_cumprod::Vector{Float32}
end

function NoiseScheduler(num_timesteps::Int, beta_start::Float32, beta_end::Float32, beta_schedule::String)
    if beta_schedule == "linear"
        betas = LinRange(beta_start, beta_end, num_timesteps)
    elseif beta_schedule == "quadratic"
        betas = LinRange(sqrt(beta_start), sqrt(beta_end), num_timesteps) .^ 2
    else
        error("Unsupported beta_schedule: $beta_schedule")
    end

    alphas = 1 .- betas
    alphas_cumprod = accumulate(*, alphas)
    return NoiseScheduler(
        num_timesteps,
        betas,
        alphas,
        alphas_cumprod,
        sqrt.(alphas_cumprod),
        sqrt.(1 .- alphas_cumprod)
    )
end

function add_noise(scheduler::NoiseScheduler, x_start, noise, t)
    s1 = scheduler.sqrt_alphas_cumprod[t]
    s2 = scheduler.sqrt_one_minus_alphas_cumprod[t]
    return @. s1 * x_start + s2 * noise
end

# Training setup
function main()
    # Argument parsing
    parser = ArgParseSettings()
    @add_arg_table parser begin
        "--experiment_name"
        help = "Name of the experiment"
        arg_type = String
        default = "base"

        "--num_epochs"
        help = "Number of training epochs"
        arg_type = Int
        default = 200

        "--train_batch_size"
        help = "Training batch size"
        arg_type = Int
        default = 32

        "--num_timesteps"
        help = "Number of timesteps for noise scheduling"
        arg_type = Int
        default = 50

        "--beta_schedule"
        help = "Beta schedule for noise (linear or quadratic)"
        arg_type = String
        default = "linear"

        "--hidden_size"
        help = "Size of hidden layers"
        arg_type = Int
        default = 128

        "--hidden_layers"
        help = "Number of hidden layers"
        arg_type = Int
        default = 3

        "--embedding_size"
        help = "Size of the embedding vectors"
        arg_type = Int
        default = 128

        "--time_embedding"
        help = "Type of time embedding (e.g., sinusoidal)"
        arg_type = String
        default = "sinusoidal"

        "--input_embedding"
        help = "Type of input embedding (e.g., sinusoidal)"
        arg_type = String
        default = "sinusoidal"
    end
    args = parse_args(parser)

    # Create dataset (dummy example here)
    x_train = randn(1000, 2)  # Replace with actual data loading
    dataset = Flux.DataLoader((x_train,), batchsize=args["train_batch_size"], shuffle=true)

    model = MLP(
        args["hidden_size"],
        args["hidden_layers"],
        args["embedding_size"],
        args["time_embedding"],
        args["input_embedding"]
    )

    scheduler = NoiseScheduler(
        args["num_timesteps"],
        0.0001f0,
        0.02f0,
        args["beta_schedule"]
    )

    opt = Flux.Adam()
    loss_fn(x, noise, t) = mean((model(add_noise(scheduler, x, noise, t), t) .- noise).^2)

    # Training loop
    for epoch in 1:args["num_epochs"]
        total_loss = 0.0
        for batch in dataset
            x = batch[1]
            noise = randn(size(x))
            t = rand(1:args["num_timesteps"], size(x, 1))
            loss_fn(x, noise, t)
            gs = gradient(() -> loss_fn(x, noise, t), Flux.params(model))
            Flux.Optimise.update!(opt, Flux.params(model), gs)
            total_loss += loss_fn(x, noise, t)
        end
        println("Epoch $epoch, Loss: $(total_loss / length(dataset))")
    end

    println("Training complete!")
end

main()
