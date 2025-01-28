using Lux

# Define a simple stateful model
struct MyStatefulLayer
    weight::Float32
end

# Make it a StatefulLuxLayer
Lux.@lux function (m::MyStatefulLayer)(x)
    state = Dict(:running_sum => 0.0)
    y = x .* m.weight
    new_state = Dict(:running_sum => state[:running_sum] + sum(y))
    return y, new_state
end

# Instantiate the model
model = MyStatefulLayer(0.5)

# Input tensor
x = rand(Float32, 4, 4)

# Forward pass with state tracking
y, updated_state = model(x)

