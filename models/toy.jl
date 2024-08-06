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
    nf = args.model.nf
    layers = Chain(
                   flatten,
                   Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    cond_layers = Chain(
                        Dense(1, nf * 4, relu),
                        Dense(nf * 4, nf * 2, relu),
                        Dense(nf * 2, nf * 1)
                       )
    linear = Dense(nf * 5, args.model.classes)
    NCClassifier(nf, layers, cond_layers, linear)
end

function (m::NCClassifier)(x, cond)
    @unpack nf, layers, cond_layers, linear = m
    cond = unsqueeze(cond, 2)
    cond = cond_layers(cond)
    x = layers(x)
    x_cond_cat = cat((x, cond), dims=2)
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
    nf = args.model.nf
    layers = Chain(
                   flatten, Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    linear = Dense(nf * 4, args.model.classes)
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
    nf = args.model.nf
    layers = Chain(
                   flatten, Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    cond_layers = Chain(
                        Dense(1, nf * 4, relu),
                        Dense(nf * 4, nf * 2, relu),
                        Dense(nf * 2, nf * 1)
                       )
    linear = Dense(nf * 5, 2)
end

function (m::NCScore)(x, cond)
    @unpack nf, layers, cond_layers, linear = m
    cond = unsqueeze(cond, 2)
    cond = cond_layers(cond)
    x = layers(x)
    x_cond_cat = cat((x, cond), dims=2)
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
    nf = args.model.nf
    layers = Chain(
                   flatten, Dense(2, nf * 16, relu),
                   Dense(nf * 16, nf * 8, relu),
                   Dense(nf * 8, nf * 4)
                  )
    linear = Dense(nf * 4, 2)
end

function (m::Score)(x)
    @unpack nf, layers, linear = m
    x = layers(x)
    out = linear(x)
    return out
end

