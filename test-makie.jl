using Plots, Makie, ImageFiltering, LinearAlgebra

x = range(-2, stop = 2, length = 21)
y = x
z = x .* exp.(-x .^ 2 .- (y') .^ 2)
scene = contour(x, y, z, levels = 10, linewidth = 3)
u, v = ImageFiltering.imgradients(z, KernelFactors.ando3)
n = vec(norm.(Vec2f0.(u,v)))
arrows!(x, y, u, v, arrowsize = n, arrowcolor = n)

# using CairoMakie
# fig = Figure()
# ax = Axis(fig[1,1])
# X = range(-2, stop=2, length=50)
# Y = range(-2, stop=2, length=50)
# f(x, y) = -x^3 + 3y - y^3
# contour!(X, Y, f, color = :blue, linewidth=2)
# dydx(x,y) = Point2f(1, x^2/(1 - y^2))
# streamplot!(dydx, -2.0..2.0, -2.0..2.0, colormap=:blues)
# fig

