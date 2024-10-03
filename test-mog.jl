using Distributions
include("lib/utils.jl")

d = MixtureModel(Normal[ Normal(-2.0, 1.2), Normal(0.0, 1.0), Normal(3.0, 2.5)], [0.2, 0.5, 0.3])

xs = -5:0.1:5
ys = pdf.(d, xs)
@> plot(xs, ys) savefig("fig/mog.pdf")

