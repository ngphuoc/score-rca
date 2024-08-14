using Enzyme
using Lux, Random, LuxCUDA

rng = Random.default_rng()
Random.seed!(rng,100)
dudt2 = Lux.Chain(x -> x.^3,
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2))
gpu_dev = gpu_device()
p, st = Lux.setup(rng, dudt2) .|> gpu_dev

function f(x::T, y::T) where T
    y .= dudt2(x, p, st)[1]
    return nothing
end

x  = [2.0f0, 2.0f0] |> gpu_dev
bx = [0.0f0, 0.0f0] |> gpu_dev
y  = [0.0f0,0.0f0] |> gpu_dev
ones32 = ones(Float32, 2) |> gpu_dev

Enzyme.autodiff(Reverse, f, Duplicated(x, bx), Duplicated(y, ones32))

