"""
A linear Gaussian CPD, always returns a Normal

	Assumes that target and all parents can be converted to Float64 (ie, are numeric)

    P(x|parents(x)) = Normal(μ=mlp(parents(x)) , σ)
"""
mutable struct NonlinearGaussianCPD <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    mlp::NamedTuple
    σ::Float64
end
NonlinearGaussianCPD(target::NodeName, μ::Float64, σ::Float64) = NonlinearGaussianCPD(target, NodeName[], Float64[], μ, σ)

name(cpd::NonlinearGaussianCPD) = cpd.target
parents(cpd::NonlinearGaussianCPD) = cpd.parents
nparams(cpd::NonlinearGaussianCPD) = length(cpd.mlp) + 2

function (cpd::NonlinearGaussianCPD)(x::Assignment)
    x = getindex.([x], cpd.parents)
    μ = cpd.mlp(x)
    Normal(μ, cpd.σ)
end

(cpd::NonlinearGaussianCPD)() = (cpd)(Assignment()) # cpd()
(cpd::NonlinearGaussianCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)

function Distributions.fit(::Type{NonlinearGaussianCPD}, data::DataFrame, target::NodeName, parents::NodeNames; args)
    H = args.H
    nparents = length(parents)
    mlp = Chain(
                Dense(nparents, H, relu),
                [Dense(H, H, relu) for _=1:args.nlayers]...,
                Dense(H, 1),
               )
    X = data[!, parents] |> Array
    y = data[!, target] |> Vector
    cpd = NonlinearGaussianCPD(target, parents, mlp, σ)
    Distributions.fit!(cpd, X, y)
end

function Distributions.fit!(cpd::NonlinearGaussianCPD, X, y; args)
    model = cpd.mlp
    opt = Flux.setup(Optimisers.AdamW(args.lr, λ=args.decay), model);
    loader = DataLoader((X, y); args.batchsize, shuffle=true)
    (x, y) = loader |> first
    @≥ x, y gpu.()

    progress = Progress(length(args.epochs))
    for epoch = 1:args.epochs
        total_loss = 0.0
        for (x, y) = loader
            @≥ x, y gpu.()
            loss, (grad,) = Flux.withgradient(model, ) do model
                Flux.mse(model(x), y)
            end
            Flux.update!(opt, model, grad)
            total_loss += loss
        end
        next!(progress; showvalues=[(:loss, total_loss/length(loader))])
    end

    return cpd
end

