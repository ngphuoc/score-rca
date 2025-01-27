using Lux, Reactant, Random, Statistics, Enzyme, MLUtils, ConcreteStructs, Printf, Optimisers, CairoMakie, MLDatasets, OneHotArrays
using Lux, ADTypes, Optimisers, Printf, Random, Reactant, Statistics, CairoMakie
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
    # train_x = Float32.(train_x) ./ 255.0 |> cdev
    train_x = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255.0 |> cdev
    train_y = onehotbatch(train_y .+ 1, 1:10) |> cdev
    return MLUtils.DataLoader((train_x, train_y), batchsize=batchsize, shuffle=true)
end

image_size=(28, 28)
num_classes=10
lr=0.001
batchsize, epochs = 128, 2
dataloader = load_mnist_lux(batchsize) |> cdev
model = LuxCNN()

rng = Random.default_rng()
Random.seed!(rng, 0)
model = LuxCNN(; input_dim=(image_size..., 1), num_classes)
ps, st = Lux.setup(rng, model) |> cdev
opt = Adam(lr)
train_state = Training.TrainState(model, ps, st, opt)
@printf "Total Trainable Parameters: %d\n" Lux.parameterlength(ps)

total_samples = 0
start_time = time()

loss_fn = CrossEntropyLoss(; logits=Val(true))
x, y = first(dataloader)
global ŷ, st = Lux.apply(model, x, ps, st)

for epoch in 1:epochs
    epoch_loss = 0.0
    for (batch_idx, (x, y)) in enumerate(dataloader)
        global total_samples += size(x, ndims(x))
        train_state = Lux.Training.TrainState(model, ps, st, Adam(0.0001f0))
        global _, loss, _, train_state = Training.single_train_step!(
            AutoEnzyme(), loss_fn, (x, y), train_state;
            return_gradients=Val(false)
        )
        isnan(loss) && error("NaN loss encountered in batch $(batch_idx) of epoch $(epoch)!")
        epoch_loss += loss
    end
    avg_loss = epoch_loss / length(dataloader)
    throughput = total_samples / (time() - start_time)
    @printf "Epoch [%2d/%2d]\tAverage Training Loss: %.6f\tThroughput: %.6f samples/s\n" epoch epochs avg_loss throughput
end

