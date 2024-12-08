using Lux, Random, Optimisers, Zygote
# using LuxCUDA, AMDGPU, Metal, oneAPI # Optional packages for GPU support

rng = Xoshiro(0)
model = Chain(BatchNorm(128), Dense(128, 256, tanh), BatchNorm(256), Chain(Dense(256, 1, tanh), Dense(1, 10)))
device = gpu_device()
ps, st = Lux.setup(rng, model) .|> device
x = rand(rng, Float32, 128, 2) |> device
y, st = Lux.apply(model, x, ps, st)
gs = only(gradient(p -> sum(first(Lux.apply(model, x, p, st))), ps))
st_opt = Optimisers.setup(Optimisers.Adam(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)

