using Enzyme, Flux

function train_enzyme!(fn, model, args...; kwargs...)
    Flux.train!(fn, Duplicated(model, Enzyme.make_zero(model)), args...; kwargs...)
end

data = [([x], 2x-x^3) for x in -2:0.1f0:2]
model = Chain(Dense(1 => 23, tanh), Dense(23 => 1, bias=false), only)
optim = Flux.setup(Adam(), model)

fn(m, x, y) = (m(x) - y)^2

@time for epoch in 1:1000
    Flux.train!(fn, model, data, optim)
end

@time for epoch in 1:1000
    Flux.train!(fn, model, data, optim)
end

@time for epoch in 1:1000
    Flux.train!(fn, model, data, optim)
end

@time for epoch in 1:1000
    train_enzyme!(fn, model, data, optim)
end

@time for epoch in 1:1000
    train_enzyme!(fn, model, data, optim)
end

@time for epoch in 1:1000
    train_enzyme!(fn, model, data, optim)
end

