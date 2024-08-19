using Enzyme, Flux, BenchmarkTools

function train_enzyme!(loss, model, dmodel, data, optim)
    @which Flux.train!(loss, Duplicated(model, dmodel), data, optim)
end

x = rand(Float32, 24, 200);
y = Matrix{Float32}(undef, 2, 200) .= randn.() .* 10;

model = Chain(Dense(24 => 8, tanh), Dense(8 => 5, relu), Dense(5 => 2), )
optim = Flux.setup(Adam(), model)
model(x)
loss(model, x, y) = mean(abs2, model(x) - y)
data = [(x, y)]

# 700 +- 600 μs
@btime for epoch in 1:10
    Flux.train!(loss, model, data, optim)
end

# 1.4 +- 900 μs
dmodel = Enzyme.make_zero(model)
@btime for epoch in 1:10
    train_enzyme!(loss, model, dmodel, data, optim)
end

