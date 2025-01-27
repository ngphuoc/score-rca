using Flux, Optimisers, MLDatasets, Random, Flux, MLUtils, Statistics
# using CUDA
dev = cpu

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
    train_x = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255.0 |> dev
    train_y = Flux.onehotbatch(train_y .+ 1, 1:10) |> dev
    return DataLoader((train_x, train_y), batchsize=batchsize, shuffle=true)
end

# Benchmark
batchsize, epochs = 128, 2
train_data = MLDatasets.MNIST()  # i.e. split=:train
test_data = MLDatasets.MNIST(split=:test)

function loader(data::MNIST=train_data; batchsize::Int=64)
    x4dim = reshape(data.features, 28,28,1,:)   # insert trivial channel dim
    yhot = Flux.onehotbatch(data.targets, 0:9)  # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true) |> dev
end

loader()  # returns a DataLoader, with first element a tuple like this:

x1, y1 = first(loader()); # (28×28×1×64 Array{Float32, 3}, 10×64 OneHotMatrix(::Vector{UInt32}))

dataloader = load_mnist_flux(batchsize)
model = FluxCNN() |> dev

y1hat = model(x1)  # try it out

settings = (;
    eta = 3e-4,     # learning rate
    lambda = 1e-2,  # for weight decay
    batchsize = 128,
    epochs = 10,
)
train_log = []

opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
opt_state = Flux.setup(opt_rule, model);

loss_fn(x, y) = Flux.logitcrossentropy(model(x), y)  # did not include softmax in the model

# @time train_flux!(model, dataloader, opt, epochs)
# loss_fn(x, y) = mean(Flux.crossentropy(model(x), y))
for epoch in 1:epochs
    @time for (x,y) in loader(; batchsize)
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
        Flux.update!(opt_state, model, grads[1])
    end
end

