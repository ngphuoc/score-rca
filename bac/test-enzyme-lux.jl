using Lux, ComponentArrays, Random, Enzyme
Enzyme.API.runtimeActivity!(true)

rng = Random.default_rng()

# Define a basic neural network structure
NN = Lux.Chain( Lux.Dense(5 => 5, tanh),
Lux.Dense(5 => 1) )

# Setup the network
ps, st = Lux.setup(rng, NN)

# Test the intialized network with some input values
x_test = [0.1, 0.2, 0.3, 0.4, 0.5]
NN(x_test, ps, st)[1][1]

dx_test = zeros(size(x_test)[1])
ax_test = getaxes( ComponentArray(ps) )
theta_test = getdata( ComponentArray(ps) ) |> f64
dtheta_test = zeros(size(theta_test)[1])

function test_function(NN, x, theta, ax, st)
y, _ = NN(x, ComponentArray(theta, ax), st)
return sum(y)
end

autodiff(Reverse, test_function, Active, Const(NN), Duplicated(x_test, dx_test), Const(theta_test), Const(ax_test), Const(st))

autodiff(Reverse, test_function, Active, Const(NN), Duplicated(x_test, dx_test), Duplicated(theta_test, dtheta_test), Const(ax_test), Const(st))

