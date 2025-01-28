using Flux, Optimisers, MLDatasets, Random, Flux, MLUtils, CUDA, Statistics

# Define CNN
function FluxCNN(; input_dim=(28, 28, 1), num_classes=10)
    return Chain(
        Conv((3, 3), input_dim[3] => 32, relu, pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu, pad=1),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(prod(input_dim[1:2] .÷ 4) * 64, 128, relu),
        Dense(128, num_classes)
    )
end

# Load MNIST
function load_mnist_flux(batchsize)
    train_x, train_y = MLDatasets.MNIST(split=:train)[:]
    train_x = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255.0 |> gpu
    train_y = Flux.onehotbatch(train_y .+ 1, 1:10) |> gpu
    return DataLoader((train_x, train_y), batchsize=batchsize, shuffle=true)
end


# Training
function train_flux!(model, dataloader, opt, epochs)
    loss_fn(x, y) = mean(Flux.crossentropy(model(x), y))
    for epoch in 1:epochs
        @time for (x, y) in dataloader
            gs = gradient(() -> loss_fn(x, y), Flux.params(model))
            opt(model, gs)
        end
    end
end

# Benchmark
batchsize, epochs = 128, 2
dataloader = load_mnist_flux(batchsize)
model = FluxCNN() |> gpu

# x, y = first(dataloader)
# model[1](x)
# model(x)
# loss_fn(x, y) = mean(Flux.crossentropy(model(x), y))

opt = Flux.Optimise.Adam(0.001)
@time train_flux!(model, dataloader, opt, epochs)

