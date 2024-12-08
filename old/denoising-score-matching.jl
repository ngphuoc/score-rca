using Flux, Functors, Optimisers

struct DSM{T}
    σ_max::Float32
    model::T
end
@functor DSM
@showfields DSM
Optimisers.trainable(dnet::DSM) = (; dnet.model)

# to pass model net from outside
# function DSM(input_dim; args)
#     X = F = n_groups = input_dim
#     H, fourier_scale = args.hidden_dim, args.fourier_scale
#     model = ConditionalChain(
#                              Parallel(.+, Dense(X, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
#                              Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
#                              Parallel(.+, Dense(H, H), Chain(RandomFourierFeatures(H, fourier_scale), Dense(H, H))), swish,
#                              Dense(H, X),
#                             )
#     return DSM(args.σ_max, model)
# end

function (dnet::DSM)(x::AbstractMatrix{T}, t) where {T}
    h = dnet.model(x, t)
    σ_t = expand_dims(marginal_prob_std(t; dnet.σ_max), 1)
    h ./ σ_t
end

function dsm_loss(dnet, x::AbstractMatrix{<:Real}; ϵ=1.0f-5, σ_max)
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1f0 - ϵ) .+ ϵ  # same t for j and paj
    z = randn!(similar(x))
    σ_t = expand_dims(marginal_prob_std(t; σ_max), 1)
    x̃ = x + z .* σ_t
    score = dnet(x̃, t)
    return sum(abs2, score .* σ_t + z) / batchsize
end

function train_dsm(dnet, X; args, ε=1f-5)
    loader = DataLoader((X,); args.batchsize, shuffle=true)
    (x,) = @> loader first args.to_device
    d = size(x, 1)
    batchsize = size(x)[end]
    t = rand!(similar(x, batchsize)) .* (1f0 - ε) .+ ε  # same t for j and paj
    dnet(x, t)
    dsm_loss(dnet, x; args.σ_max)
    # eval_unet(dnet, df)
    opt = Flux.setup(Optimisers.Adam(args.lr), dnet);
    progress = Progress(args.epochs, desc="Fitting dnet")
    for epoch = 1:args.epochs
        total_loss = 0.0
        for (x,) = loader
            @≥ x args.to_device
            global loss, (grad,) = Flux.withgradient(dnet, ) do dnet
                dsm_loss(dnet, x; args.σ_max)
            end
            Flux.update!(opt, dnet, grad)
            total_loss += loss
        end
        next!(progress; showvalues=[(:loss, total_loss/length(loader))])
    end
    return dnet
    # @≥ X, dnet cpu.();
    # BSON.@save "data/main-rca.bson" args X dnet
    # BSON.@load "data/main-rca.bson" args X dnet
    # @≥ X, dnet to_device.();
end

