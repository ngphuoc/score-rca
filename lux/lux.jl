using Lux, Reactant, Random, Statistics, Enzyme, MLUtils, ConcreteStructs, Printf, Optimisers, CairoMakie, MLDatasets, OneHotArrays
const xdev = reactant_device(; force=true)
const cdev = cpu_device()

# Define CNN
function LuxCNN(; input_dim=(28, 28, 1), num_classes=10)
    return Chain(
        Conv((3, 3), input_dim[3] => 32, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu; pad=1),
        MaxPool((2, 2)),
        FlattenLayer(),
        Dense(prod(input_dim[1:2] .÷ 4) * 64, 128, relu),
        Dense(128, num_classes)
    )
end

# Load MNIST
function load_mnist_lux(batchsize)
    train_x, train_y = MLDatasets.MNIST(split=:train)[:]
    train_x = Float32.(train_x) ./ 255.0 |> xdev
    train_y = onehotbatch(train_y .+ 1, 1:10) |> xdev
    return MLUtils.DataLoader((train_x, train_y), batchsize=batchsize, shuffle=true)
end

# Training
function train_lux!(model, dataloader, opt, epochs)
    ps, st = Lux.setup(Random.default_rng(), model) |> xdev
    loss_fn = (ps, st, x, y) -> mean(crossentropy(model(x, ps, st)[1], y))
    for epoch in 1:epochs
        @time for (x, y) in dataloader
            gs = gradient(ps) do ps
                loss_fn(ps, st, x, y)
            end
            Optimisers.update!(opt, ps, gs)
        end
    end
end

# Benchmark
batchsize, epochs = 128, 2
dataloader = load_mnist_lux(batchsize)
model = LuxCNN()
opt = Optimisers.Adam(0.001)
@time train_lux!(model, dataloader, opt, epochs)

