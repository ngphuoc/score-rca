using Flux, Zygote

# Define a function f: ℝ^n → ℝ^m.
f(x) = [x[1]^2 + sin(x[2]), x[1]*x[2], exp(x[1]) - x[2]]

# Function to compute the Jacobian of f at x
function jacobian(f, x)
    # Evaluate f and get the pullback function.
    y, back = Zygote.pullback(f, x)
    # Initialize a Jacobian matrix: rows = length(y), columns = length(x)
    J = zeros(eltype(x), length(y), length(x))
    # For each output element, compute the gradient vector (row of the Jacobian)
    for i in 1:length(y)
        # Create a one-hot vector (seed) for the i-th output.
        seed = zeros(eltype(y), length(y))
        seed[i] = one(eltype(y))
        # back(seed) returns a tuple; here we have a single input `x`
        J[i, :] = back(seed)[1]
    end
    return J
end

# Example usage:
x = [1.0, 2.0]
J = jacobian(f, x)
println("Function output: ", f(x))
println("Jacobian:\n", J)

Flux.jacobian(f, x)[1]

