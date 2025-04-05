using PyPlot

function meshgrid(x, y)
    X = repeat(reshape(collect(x), 1, length(x)), length(y), 1)
    Y = repeat(reshape(collect(y), length(y), 1), 1, length(x))
    return X, Y
end

# Define grid boundaries and resolution
x = LinRange(-5, 5, 20)
y = LinRange(-5, 5, 20)

# Use your meshgrid function
X, Y = meshgrid(x, y)
# Compute the gradient field of f(x, y) = x^2 + y^2
U = 2 .* X   # ∂f/∂x = 2x
V = 2 .* Y   # ∂f/∂y = 2y

# Create the quiver plot
figure(figsize=(8, 6))
quiver(X, Y, U, V, color="r", angles="xy", scale_units="xy", scale=1.5)
title("Gradient Field of f(x,y) = x^2 + y^2")
xlabel("x")
ylabel("y")
xlim(-5, 5)
ylim(-5, 5)
grid(true)
savefig("fig/quiver.png")
