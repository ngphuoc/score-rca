using MLDatasets, Flux, JLD2
dev = cpu

folder = "runs"  # sub-directory in which to save
isdir(folder) || mkdir(folder)
filename = joinpath(folder, "lenet.jld2")

train_data = MLDatasets.MNIST()  # i.e. split=:train
test_data = MLDatasets.MNIST(split=:test)

function loader(data::MNIST=train_data; batchsize::Int=64)
    x4dim = reshape(data.features, 28,28,1,:)   # insert trivial channel dim
    yhot = Flux.onehotbatch(data.targets, 0:9)  # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true) |> dev
end

loader()  # returns a DataLoader, with first element a tuple like this:

x1, y1 = first(loader()); # (28×28×1×64 Array{Float32, 3}, 10×64 OneHotMatrix(::Vector{UInt32}))

lenet = Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
) |> dev

y1hat = lenet(x1)  # try it out

sum(softmax(y1hat); dims=1)

@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

using Statistics: mean  # standard library

function loss_and_accuracy(model, data::MNIST=test_data)
    (x,y) = only(loader(data; batchsize=length(data)))  # make one big batch
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    (; loss, acc, split=data.split)  # return a NamedTuple
end

@show loss_and_accuracy(lenet);  # accuracy about 10%, before training

settings = (;
    eta = 3e-4,     # learning rate
    lambda = 1e-2,  # for weight decay
    batchsize = 128,
    epochs = 10,
)
train_log = []

opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
opt_state = Flux.setup(opt_rule, lenet);

for epoch in 1:settings.epochs
    # @time will show a much longer time for the first epoch, due to compilation
    @time for (x,y) in loader(batchsize=settings.batchsize)
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), lenet)
        Flux.update!(opt_state, lenet, grads[1])
    end

    # Logging & saving, but not on every epoch
    if epoch % 2 == 1
        loss, acc, _ = loss_and_accuracy(lenet)
        test_loss, test_acc, _ = loss_and_accuracy(lenet, test_data)
        @info "logging:" epoch acc test_acc
        nt = (; epoch, loss, acc, test_loss, test_acc)  # make a NamedTuple
        push!(train_log, nt)
    end
    if epoch % 5 == 0
        JLD2.jldsave(filename; lenet_state = Flux.state(lenet) |> cpu)
        println("saved to ", filename, " after ", epoch, " epochs")
    end
end

@show train_log;

y1hat = lenet(x1)
@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

