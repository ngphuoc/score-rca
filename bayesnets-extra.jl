
@showfields BayesNet
@showfields NonlinearGaussianCPD
@showfields NonlinearScoreCPD

function Distributions.fit(::Type{NonlinearGaussianCPD}, df::DataFrame, node::NodeName, pa::NodeNames; args)
    H = args.hidden_dim
    nparents = length(pa)
    mlp = Chain(
                Dense(nparents, H, relu),
                [Dense(H, H, relu) for _=1:args.n_layers-2]...,
                Dense(H, 1),
               )
    X = df[!, pa] |> Array;
    Y = df[!, node] |> Vector;
    cpd = NonlinearGaussianCPD(node, pa, mlp, std(Y)/nparents)
    model = cpd.mlp |> gpu;
    opt = Flux.setup(Optimisers.AdamW(args.lr, (0.9, 0.999), args.decay), model);
    # AdamW(η = 0.001, β = (0.9, 0.999), λ = 0, ϵ = 1e-8)
    loader = DataLoader((X', Y'); args.batchsize, shuffle=true)
    (x, y) = loader |> first
    @≥ x, y gpu.()
    Flux.mse(model(x), y)
    progress = Progress(args.epochs, desc="Fitting FCM for node $(name(cpd))")
    for epoch = 1:args.epochs
        total_loss = 0.0
        for (x, y) = loader
            @≥ x, y gpu.()
            # loss = Flux.mse(model(x), y)
            loss, (grad,) = Flux.withgradient(model, ) do model
                Flux.mse(model(x), y)
            end
            Flux.update!(opt, model, grad)
            total_loss += loss
        end
        next!(progress; showvalues=[(:loss, total_loss/length(loader))])
    end
    cpd.mlp = cpu(model)
    return cpd
end

function Distributions.fit(::Type{NonlinearScoreCPD}, df::DataFrame, node::NodeName, pa::NodeNames; args)
    H = args.hidden_dim
    nparents = length(pa)
    mlp = Chain(
                Dense(nparents, H, relu),
                [Dense(H, H, relu) for _=1:args.n_layers-2]...,
                Dense(H, 1),
               )
    score = MlpUnet(nparents + 1) |> gpu
    X = df[!, pa] |> Array;
    Y = df[!, node] |> Vector;
    cpd = NonlinearScoreCPD(node, pa, mlp, score, std(Y)/nparents)
    model = cpd.mlp |> gpu;
    opt = Flux.setup(Optimisers.AdamW(args.lr, (0.9, 0.999), args.decay), model);
    # AdamW(η = 0.001, β = (0.9, 0.999), λ = 0, ϵ = 1e-8)
    loader = DataLoader((X', Y'); args.batchsize, shuffle=true)
    (x, y) = loader |> first
    @≥ x, y gpu.()
    Flux.mse(model(x), y)
    progress = Progress(args.epochs, desc="Fitting FCM for node $(name(cpd))")
    for epoch = 1:args.epochs
        total_loss = 0.0
        for (x, y) = loader
            @≥ x, y gpu.()
            # loss = Flux.mse(model(x), y)
            loss, (grad,) = Flux.withgradient(model, ) do model
                Flux.mse(model(x), y)
            end
            Flux.update!(opt, model, grad)
            total_loss += loss
        end
        next!(progress; showvalues=[(:loss, total_loss/length(loader))])
    end
    cpd.mlp = cpu(model)
    return cpd
end

