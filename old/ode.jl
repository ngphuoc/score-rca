# ]
add Lux LuxCUDA ComponentArrays Zygote Plots DiffEqFlux

using DiffEqFlux
using Lux, LuxCUDA

using OrdinaryDiffEq, Lux, LuxCUDA, SciMLSensitivity, ComponentArrays, Randomode
using DiffEqFlux: NeuralODE
using Optimization, OptimizationOptimisers, Zygote, Plots

# Ensure GPU scalar operations are avoided for performance
CUDA.allowscalar(false)

# Set up the random generator and devices
rng = Xoshiro(0)
const cdev = cpu_device()
const gdev = gpu_device()

# Define the model
model = Chain(Dense(2, 50, tanh), Dense(50, 2))
ps, st = Lux.setup(rng, model)
ps = ps |> ComponentArray |> gdev
st = st |> gdev

# Define the neural ODE
dudt(u, p, t) = model(u, p, st)[1]

# Simulation interval and intermediary points
tspan = (0.0f0, 10.0f0)
tsteps = 0.0f0:0.1f0:10.0f0

# Initial condition on the GPU
u0 = Float32[2.0; 0.0] |> gdev
prob_gpu = ODEProblem(dudt, u0, tspan, ps)

# Solve the ODE on GPU
sol_gpu = solve(prob_gpu, Tsit5(); saveat = tsteps)

# Training a Neural ODE
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

# Generate data
tspan_data = (0.0f0, 1.5f0)
tsteps_data = range(tspan_data[1], tspan_data[2]; length = 30)
prob_trueode = ODEProblem(trueODEfunc, u0, tspan_data)
ode_data = solve(prob_trueode, Tsit5(); saveat = tsteps_data) |> Array |> gdev

# Define a second neural ODE for training
dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
p2, st2 = Lux.setup(rng, dudt2)
p2 = p2 |> ComponentArray |> gdev
st2 = st2 |> gdev

prob_neuralode = NeuralODE(dudt2, tspan_data, Tsit5(); saveat = tsteps_data)

# Prediction and loss functions
predict_neuralode(p) = reduce(hcat, first(prob_neuralode(u0, p, st2)).u)
function loss_neuralode(p)
    pred = predict_neuralode(p)
    return sum(abs2, ode_data .- pred), pred
end

# Callback for monitoring training
list_plots = []
callback = function (p, l, pred; doplot = false)
    global list_plots
    display(l)
    plt = scatter(tsteps_data, Array(ode_data[1, :]); label = "data")
    scatter!(plt, tsteps_data, Array(pred[1, :]); label = "prediction")
    push!(list_plots, plt)
    if doplot
        display(plot(plt))
    end
    return false
end

# Set up and solve the optimization problem
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p2)
result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback, maxiters = 300
)

# Visualize results
final_pred = predict_neuralode(result_neuralode.u)
scatter(tsteps_data, Array(ode_data[1, :]); label = "data", title = "Neural ODE Training")
scatter!(tsteps_data, Array(final_pred[1, :]); label = "prediction")

