using Lux, ADTypes, Optimisers, Printf, Random, Reactant, Statistics, CairoMakie

function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, Float32, (1, 128)) .* 0.1f0
    return (x, y)
end

rng = MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3, color=:blue)
    s = scatter!(ax, x[1, :], y[1, :]; markersize=12, alpha=0.5,
        color=:orange, strokecolor=:black, strokewidth=2)

    axislegend(ax, [l, s], ["True Quadratic Function", "Data Points"])

    fig
end

model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

opt = Adam(0.03f0)

const loss_function = MSELoss()

const cdev = cpu_device()
const xdev = reactant_device()

ps, st = Lux.setup(rng, model) |> xdev

tstate = Training.TrainState(model, ps, st, opt)

vjp_rule = AutoEnzyme()

function main(tstate::Training.TrainState, vjp, data, epochs)
    data = data |> xdev
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)

forward_pass = @compile Lux.apply(
    tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states)
)

y_pred = cdev(first(forward_pass(
    tstate.model, xdev(x), tstate.parameters, Lux.testmode(tstate.states)
)))
nothing #hide

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    l = lines!(ax, x[1, :], x -> evalpoly(x, (0, -2, 1)); linewidth=3)
    s1 = scatter!(ax, x[1, :], y[1, :]; markersize=12, alpha=0.5,
        color=:orange, strokecolor=:black, strokewidth=2)
    s2 = scatter!(ax, x[1, :], y_pred[1, :]; markersize=12, alpha=0.5,
        color=:green, strokecolor=:black, strokewidth=2)

    axislegend(ax, [l, s1, s2], ["True Quadratic Function", "Actual Data", "Predictions"])

    fig
end
