using BayesNets
using BayesNets: NonlinearGaussianCPD

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
    for _ = 1:args.epochs
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
    for _ = 1:args.epochs
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

function Distributions.fit!(bn::BayesNet, data::AbstractArray, )
    d = nv(bn.dag)
    js = 1:d
    adj = @> bn.dag adjacency_matrix Matrix{Bool}
    pas =  getindex.([adj], :, js)
    for j = 1:d
        Distributions.fit!(bn.cpds[j], data, j, pas[j])
    end
    nothing
end

function Distributions.fit!(cpd::RootCPD, data::AbstractArray, target::Int, parents; min_stdev::Float64=0.0, )
    x = data[target, :]
    μ = convert(Float64, mean(x))
    σ = convert(Float64, stdm(x, μ))
    σ = max(σ, min_stdev)
    cpd.d = Normal(μ, σ)
    return nothing
end

""" solve the regression problem
  β = (XᵀX)⁻¹Xᵀy
    X is the [nsamples × nparents+1] data matrix
    where the last column is 1.0
    y is the [nsamples] vector of target values
NOTE: this will fail if X is not full rank
"""
function Distributions.fit!(cpd::LinearCPD, data::AbstractArray, target::Int, parents; min_stdev=0.0,)
    nparents = sum(parents .!= 0)
    n = size(data, 2)
    X = vcat(data[parents, :], ones(1, n))
    y = data[target, :]
    β = (X*X')\(X*y)
    b = β[end]
    σ = max(std(y), min_stdev)
    cpd.d = Normal(b, σ)
    cpd.a = β[1:nparents]
    nothing
end

function Distributions.fit!(cpd::LinearBayesianCPD, data::AbstractArray, target::Int, parents; min_stdev=0.0,)
    nparents = sum(parents .!= 0)
    n = size(data, 2)
    X = @> data[parents, :]' f64
    y = @> data[target, :] f64
    Distributions.fit!(cpd::LinearBayesianCPD, X, y)
    nothing
end

