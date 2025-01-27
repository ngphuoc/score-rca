using Lux, Reactant, Random, Statistics, Enzyme, MLUtils, ConcreteStructs, Printf, Optimisers, CairoMakie, MLDatasets, OneHotArrays
using Lux, ADTypes, Optimisers, Printf, Random, Reactant, Statistics, CairoMakie
# const xdev = reactant_device(; force=true)
using Metal
const xdev = gpu_device()
const cdev = cpu_device()
include("./lib/utils.jl")

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

function loadmnist(; batchsize, train_split)
    ## Load MNIST
    dataset = MNIST(; split=:train)
    imgs = dataset.features
    labels_raw = dataset.targets
    ## Process images into (H, W, C, BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = @> onehotbatch(labels_raw, 0:9) Array{Float32}
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)
    return (
        DataLoader((x_train, y_train); batchsize, shuffle=true, partial=false),
        DataLoader((x_test, y_test); batchsize, shuffle=false, partial=false)
    )
end

function main()
    image_size=(28, 28)
    num_classes=10
    lr=0.001
    batchsize = 128
    epochs = 2
    train_split = 0.8
    dataloader, testloader = loadmnist(; batchsize, train_split)
    x, y = @> first(dataloader) xdev

    rng = Random.default_rng()
    Random.seed!(rng, 0)
    model = LuxCNN(; input_dim=(image_size..., 1), num_classes)
    ps, st = @> Lux.setup(rng, model) xdev
    opt = Adam(lr)
    train_state = Training.TrainState(model, ps, st, opt)
    @printf "Total Trainable Parameters: %d\n" Lux.parameterlength(ps)

    total_samples = 0
    start_time = time()

    loss_fn = CrossEntropyLoss(; logits=Val(true))
    ŷ, st = Lux.apply(model, x, ps, st)

    for epoch in 1:epochs
        epoch_loss = 0.0
        for (batch_idx, (x, y)) = enumerate(dataloader)
            x, y = (x, y) |> xdev
            total_samples += size(x, ndims(x))
            _, loss, _, train_state = Training.single_train_step!( AutoEnzyme(), loss_fn, (x, y), train_state; return_gradients=Val(false))
            isnan(loss) && error("NaN loss encountered in batch $(batch_idx) of epoch $(epoch)!")
            epoch_loss += loss
        end
        avg_loss = epoch_loss / length(dataloader)
        throughput = total_samples / (time() - start_time)
        @printf "Epoch [%2d/%2d]\tAverage Training Loss: %.6f\tThroughput: %.6f samples/s\n" epoch epochs avg_loss throughput
    end
end

main()

