
@info "Plots"
using LaTeXStrings
include("lib/plot-utils.jl")
xlim = ylim = (-5, 5)

#-- defaults
default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

#-- plot data
x, y = eachrow(X_val)
pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

#-- plot perturbed data
x = @> X_val gpu;
d = size(x, 1)
X, batchsize = size(x, 1), size(x)[end]
t = rand!(similar(x, size(x)[end])) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
z = 2rand!(similar(x)) .- 1;
x̃ = x .+ σ_t .* z
x, y = eachrow(cpu(x̃))
pl_sm_data = scatter(x, y; xlab=L"x", ylab=L"y", title="Perturbed score matching")

#-- plot gradients
μx, σx = @> X_val mean(dims=2), std(dims=2)
xlim = ylim = (-2, 2)
x = @> Iterators.product(range(xlim..., length=20), range(ylim..., length=20)) collect vec;
x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
d = size(x, 1)
t = fill!(similar(x, size(x)[end]), 0.1) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
J = @> net(x, t)
@≥ J, x cpu.()
x, y = eachrow(x);
u, v = eachrow(0.2J);
pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
arrow0!.(x, y, u, v; as=0.2, lw=1.0);

@> Plots.plot(pl_data, pl_sm_data, pl_gradient; xlim, ylim, size=(1000, 800)) savefig("fig/spiral-2d.png")

