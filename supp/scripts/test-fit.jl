using Plots
t = 0:0.2:40
y = @. 1 - 6e-3 * (t/40 - 1/(1+0.5t)*sin(2π*t/6))
b = 1.0
a = t \ (y .- b)
scatter(t, y, ms=2)
plot!(t, a*t .+ b, c =:red)
