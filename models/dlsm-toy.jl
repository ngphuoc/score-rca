using Flux
using Flux: @functor
using Parameters: @unpack

struct NCClassifier
    nf
    layers
    cond_layers
    linear
end
@functor NCClassifier layers, cond_layers, linear


"""noise-conditioned classifie"""

function NCClassifier(; args)
    nf = args.nf
    layers = Chain(
                   Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    cond_layers = Chain(
                        Dense(1, nf * 4, relu),
                        Dense(nf * 4, nf * 2, relu),
                        Dense(nf * 2, nf * 1)
                       )
    linear = Dense(nf * 5, args.classes)
    NCClassifier(nf, layers, cond_layers, linear)
end

function (m::NCClassifier)(x, σ_cond)
    @unpack nf, layers, cond_layers, linear = m
    σ_cond = cond_layers(σ_cond)
    x = layers(x)
    x_cond_cat = cat(x, σ_cond, dims=1)
    out = linear(x_cond_cat)
    return out
end

struct Classifier
    nf
    layers
    linear
end
@functor Classifier layers, linear

"""simple classifie"""
function Classifier(; args)
    nf = args.nf
    layers = Chain(
                   Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    linear = Dense(nf * 4, args.classes)
    Classifier(nf, layers, linear)
end

function (m::NCClassifier)(x)
    @unpack nf, layers, linear = m
    x = layers(x)
    out = linear(x)
    return out
end


struct NCScore
    nf
    layers
    cond_layers
    linear
end
@functor NCScore layers, cond_layers, linear

"""noise-conditioned score model"""
function NCScore(; args)
    nf = args.nf
    layers = Chain(
                   Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    cond_layers = Chain(
                        Dense(1, nf * 4, relu),
                        Dense(nf * 4, nf * 2, relu),
                        Dense(nf * 2, nf * 1)
                       )
    linear = Dense(nf * 5, 2)
    NCScore(nf, layers, cond_layers, linear)
end

function (m::NCScore)(x, σ_cond)
    @unpack nf, layers, cond_layers, linear = m
    σ_cond = cond_layers(σ_cond)
    x = layers(x)
    x_cond_cat = cat(x, σ_cond, dims=1)
    out = linear(x_cond_cat)
    return out
end

struct Score
    nf
    layers
    linear
end
@functor Score layers, linear

"""simple score model"""
function Score(args)
    @unpack nf, layers, linear = m
    nf = args.nf
    layers = Chain(
                   Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    linear = Dense(nf * 4, 2)
    Score(nf, layers, linear)
end

function (m::Score)(x)
    @unpack nf, layers, linear = m
    x = layers(x)
    out = linear(x)
    return out
end

