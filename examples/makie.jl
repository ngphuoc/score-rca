using CairoMakie

# Create a grid of points
x = 1:10
y = 1:10
X, Y = rand(10), rand(10)

# Define the vector field
U = cos.(X)
V = sin.(Y)

# Create the quiver plot
fig = Figure()
Axis(fig[1, 1])
quiver!(X, Y, U, V)

# Display the plot
fig
