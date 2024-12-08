using Pkg
Pkg.add("CairoMakie")
Pkg.add("GLMakie")

using GLMakie

# Define the grid
x = range(-2, 2, length=20)
y = range(-2, 2, length=20)
X, Y = meshgrid(x, y)

# Define the gradient field
U = -Y
V = X

# Create the quiver plot
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1], title = "Gradient Field")
quiver!(ax, X, Y, U, V, color = :blue, scale = 0.1)

# Display the plot
fig
